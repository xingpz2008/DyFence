#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

#include "exec/MVZKNetworkManager.h"

struct DelayedChannel {
    std::mutex mutex;
    std::condition_variable condition;
    std::deque<std::uint8_t> prover_to_verifier;
    std::deque<std::uint8_t> verifier_to_prover;
};

class DelayedIO {
public:
    std::uint64_t counter = 0;

    DelayedIO(std::shared_ptr<DelayedChannel> channel, bool prover_side, int receive_delay_ms)
        : channel_(std::move(channel)), prover_side_(prover_side), receive_delay_ms_(receive_delay_ms) {}

    void send_data(const void* source, std::size_t length) {
        const auto* bytes = static_cast<const std::uint8_t*>(source);
        std::lock_guard<std::mutex> lock(channel_->mutex);
        auto& queue = prover_side_ ? channel_->prover_to_verifier : channel_->verifier_to_prover;
        queue.insert(queue.end(), bytes, bytes + length);
        counter += length;
        channel_->condition.notify_all();
    }

    void recv_data(void* destination, std::size_t length) {
        if (receive_delay_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(receive_delay_ms_));
        }
        auto* bytes = static_cast<std::uint8_t*>(destination);
        std::unique_lock<std::mutex> lock(channel_->mutex);
        auto& queue = prover_side_ ? channel_->verifier_to_prover : channel_->prover_to_verifier;
        channel_->condition.wait(lock, [&queue, length] { return queue.size() >= length; });
        for (std::size_t index = 0; index < length; ++index) {
            bytes[index] = queue.front();
            queue.pop_front();
        }
    }

    void flush() {}

private:
    std::shared_ptr<DelayedChannel> channel_;
    bool prover_side_;
    int receive_delay_ms_;
};

int main() {
    auto channel_two = std::make_shared<DelayedChannel>();
    auto channel_three = std::make_shared<DelayedChannel>();
    auto* prover_io_two = new DelayedIO(channel_two, true, 0);
    auto* verifier_io_two = new DelayedIO(channel_two, false, 0);
    auto* prover_io_three = new DelayedIO(channel_three, true, 0);
    auto* verifier_io_three = new DelayedIO(channel_three, false, 500);

    MVZKNetworkManager<DelayedIO> prover;
    prover.my_party_id = emp::ALICE;
    prover.num_parties = 3;
    prover.num_vole_ios = 0;
    prover.broadcast_ios = new DelayedIO*[4]{nullptr, nullptr, prover_io_two, prover_io_three};

    MVZKNetworkManager<DelayedIO> verifier_two;
    verifier_two.my_party_id = emp::BOB;
    verifier_two.num_parties = 3;
    verifier_two.num_vole_ios = 0;
    verifier_two.broadcast_ios = new DelayedIO*[4]{nullptr, verifier_io_two, nullptr, nullptr};

    MVZKNetworkManager<DelayedIO> verifier_three;
    verifier_three.my_party_id = 3;
    verifier_three.num_parties = 3;
    verifier_three.num_vole_ios = 0;
    verifier_three.broadcast_ios = new DelayedIO*[4]{nullptr, verifier_io_three, nullptr, nullptr};

    std::atomic<bool> prover_finished{false};
    std::atomic<int> verifiers_finished{0};

    std::thread verifier_two_thread([&] {
        verifier_two.sync_network();
        ++verifiers_finished;
    });
    std::thread verifier_three_thread([&] {
        verifier_three.sync_network();
        ++verifiers_finished;
    });
    std::thread prover_thread([&] {
        prover.sync_network();
        prover_finished = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    const bool prover_returned_before_verifier_ack = prover_finished.load();

    prover_thread.join();
    verifier_two_thread.join();
    verifier_three_thread.join();

    if (prover_returned_before_verifier_ack) {
        std::cerr << "prover returned before the delayed verifier acknowledged the barrier\n";
        return 1;
    }
    if (!prover_finished || verifiers_finished != 2) {
        std::cerr << "barrier participants did not finish\n";
        return 1;
    }

    std::cout << "strict synchronization barrier passed\n";
    return 0;
}
