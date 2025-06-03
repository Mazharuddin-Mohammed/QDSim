#pragma once

#include <atomic>
#include <memory>
#include <functional>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <thread>

namespace QDSim {

template<typename Resource>
class ThreadSafeResourceManager {
public:
    using ResourcePtr = std::shared_ptr<Resource>;
    using ResourceFactory = std::function<ResourcePtr()>;
    using ResourceDeleter = std::function<void(ResourcePtr)>;
    
    ThreadSafeResourceManager(ResourceFactory factory, 
                             ResourceDeleter deleter = nullptr,
                             size_t max_resources = 0)
        : factory_(factory), deleter_(deleter), max_resources_(max_resources) {}
    
    ~ThreadSafeResourceManager() {
        cleanup();
    }
    
    // RAII wrapper for automatic resource management
    class ResourceGuard {
    public:
        ResourceGuard(ResourcePtr resource, ResourceDeleter deleter, 
                     ThreadSafeResourceManager* manager)
            : resource_(resource), deleter_(deleter), manager_(manager) {}
        
        ~ResourceGuard() {
            if (resource_) {
                if (deleter_) {
                    deleter_(resource_);
                }
                if (manager_) {
                    manager_->returnResource(resource_);
                }
            }
        }
        
        // Disable copy
        ResourceGuard(const ResourceGuard&) = delete;
        ResourceGuard& operator=(const ResourceGuard&) = delete;
        
        // Enable move
        ResourceGuard(ResourceGuard&& other) noexcept
            : resource_(std::move(other.resource_)), 
              deleter_(std::move(other.deleter_)),
              manager_(other.manager_) {
            other.resource_ = nullptr;
            other.manager_ = nullptr;
        }
        
        ResourceGuard& operator=(ResourceGuard&& other) noexcept {
            if (this != &other) {
                if (resource_ && deleter_) {
                    deleter_(resource_);
                }
                resource_ = std::move(other.resource_);
                deleter_ = std::move(other.deleter_);
                manager_ = other.manager_;
                other.resource_ = nullptr;
                other.manager_ = nullptr;
            }
            return *this;
        }
        
        Resource* operator->() const { return resource_.get(); }
        Resource& operator*() const { return *resource_; }
        Resource* get() const { return resource_.get(); }
        bool valid() const { return resource_ != nullptr; }
        
        // Release ownership (resource won't be returned to pool)
        ResourcePtr release() {
            manager_ = nullptr;
            return std::move(resource_);
        }
        
    private:
        ResourcePtr resource_;
        ResourceDeleter deleter_;
        ThreadSafeResourceManager* manager_;
    };
    
    ResourceGuard acquire() {
        ResourcePtr resource = getResource();
        return ResourceGuard(resource, deleter_, this);
    }
    
    // Get current pool statistics
    struct PoolStats {
        size_t available_resources;
        size_t total_created;
        size_t active_resources;
        size_t max_resources;
    };
    
    PoolStats getStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return {
            available_resources_.size(),
            total_created_.load(),
            active_resources_.load(),
            max_resources_
        };
    }
    
    void cleanup() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Clear available resources
        while (!available_resources_.empty()) {
            auto resource = available_resources_.front();
            available_resources_.pop();
            if (deleter_) {
                deleter_(resource);
            }
        }
        
        total_created_.store(0);
        active_resources_.store(0);
    }
    
private:
    ResourceFactory factory_;
    ResourceDeleter deleter_;
    size_t max_resources_;
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<ResourcePtr> available_resources_;
    
    std::atomic<size_t> total_created_{0};
    std::atomic<size_t> active_resources_{0};
    
    ResourcePtr getResource() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Try to get from pool first
        if (!available_resources_.empty()) {
            auto resource = available_resources_.front();
            available_resources_.pop();
            active_resources_.fetch_add(1);
            return resource;
        }
        
        // Check if we can create a new resource
        if (max_resources_ == 0 || total_created_.load() < max_resources_) {
            lock.unlock(); // Release lock during resource creation
            
            auto resource = factory_();
            if (resource) {
                total_created_.fetch_add(1);
                active_resources_.fetch_add(1);
                return resource;
            }
        }
        
        // Wait for a resource to become available
        lock.lock();
        cv_.wait(lock, [this]() { return !available_resources_.empty(); });
        
        auto resource = available_resources_.front();
        available_resources_.pop();
        active_resources_.fetch_add(1);
        return resource;
    }
    
    void returnResource(ResourcePtr resource) {
        if (!resource) return;
        
        std::lock_guard<std::mutex> lock(mutex_);
        available_resources_.push(resource);
        active_resources_.fetch_sub(1);
        cv_.notify_one();
    }
    
    friend class ResourceGuard;
};

// Lock-free queue implementation for high-performance scenarios
template<typename T>
class LockFreeQueue {
private:
    struct Node {
        std::atomic<T*> data{nullptr};
        std::atomic<Node*> next{nullptr};
    };
    
    std::atomic<Node*> head_{new Node};
    std::atomic<Node*> tail_{head_.load()};
    
public:
    LockFreeQueue() = default;
    
    ~LockFreeQueue() {
        while (Node* old_head = head_.load()) {
            head_.store(old_head->next.load());
            delete old_head;
        }
    }
    
    void enqueue(T item) {
        Node* new_node = new Node;
        T* data = new T(std::move(item));
        
        Node* prev_tail = tail_.exchange(new_node);
        prev_tail->data.store(data);
        prev_tail->next.store(new_node);
    }
    
    bool dequeue(T& result) {
        Node* head = head_.load();
        Node* next = head->next.load();
        
        if (next == nullptr) {
            return false; // Queue is empty
        }
        
        T* data = next->data.exchange(nullptr);
        if (data == nullptr) {
            return false; // Another thread got this item
        }
        
        result = *data;
        delete data;
        
        head_.store(next);
        delete head;
        
        return true;
    }
    
    bool empty() const {
        Node* head = head_.load();
        Node* next = head->next.load();
        return next == nullptr;
    }
    
    // Disable copy and move
    LockFreeQueue(const LockFreeQueue&) = delete;
    LockFreeQueue& operator=(const LockFreeQueue&) = delete;
    LockFreeQueue(LockFreeQueue&&) = delete;
    LockFreeQueue& operator=(LockFreeQueue&&) = delete;
};

} // namespace QDSim
