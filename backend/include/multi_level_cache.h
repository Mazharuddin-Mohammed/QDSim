#pragma once
/**
 * @file multi_level_cache.h
 * @brief Defines a multi-level cache system for QDSim.
 *
 * This file contains declarations of a multi-level cache system
 * for improving memory access performance in large-scale quantum simulations.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include <unordered_map>
#include <list>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <functional>
#include <chrono>
#include <atomic>
#include <any>

namespace Cache {

/**
 * @enum CacheLevel
 * @brief Enumeration of cache levels.
 */
enum class CacheLevel {
    L1,     ///< Level 1 cache (fastest, smallest)
    L2,     ///< Level 2 cache
    L3,     ///< Level 3 cache (slowest, largest)
    ALL     ///< All cache levels
};

/**
 * @enum CachePolicy
 * @brief Enumeration of cache replacement policies.
 */
enum class CachePolicy {
    LRU,    ///< Least Recently Used
    LFU,    ///< Least Frequently Used
    FIFO,   ///< First In First Out
    RANDOM  ///< Random replacement
};

/**
 * @struct CacheStats
 * @brief Structure to hold cache statistics.
 */
struct CacheStats {
    size_t capacity;            ///< Cache capacity in bytes
    size_t size;                ///< Current cache size in bytes
    size_t item_count;          ///< Number of items in the cache
    size_t hits;                ///< Number of cache hits
    size_t misses;              ///< Number of cache misses
    size_t evictions;           ///< Number of cache evictions
    double hit_ratio;           ///< Cache hit ratio (hits / (hits + misses))
    double avg_lookup_time;     ///< Average lookup time in microseconds
    double avg_insert_time;     ///< Average insert time in microseconds
};

/**
 * @class CacheItem
 * @brief Class for a cache item.
 *
 * The CacheItem class represents an item in the cache.
 * It stores the key, value, size, and metadata for the item.
 *
 * @tparam Key The key type
 * @tparam Value The value type
 */
template <typename Key, typename Value>
class CacheItem {
public:
    /**
     * @brief Constructs a new CacheItem object.
     *
     * @param key The key
     * @param value The value
     * @param size The size of the item in bytes
     */
    CacheItem(const Key& key, const Value& value, size_t size)
        : key_(key), value_(value), size_(size), access_count_(0), last_access_time_(std::chrono::steady_clock::now()) {}

    /**
     * @brief Gets the key.
     *
     * @return The key
     */
    const Key& getKey() const {
        return key_;
    }

    /**
     * @brief Gets the value.
     *
     * @return The value
     */
    const Value& getValue() const {
        return value_;
    }

    /**
     * @brief Gets the size of the item.
     *
     * @return The size in bytes
     */
    size_t getSize() const {
        return size_;
    }

    /**
     * @brief Gets the access count.
     *
     * @return The access count
     */
    size_t getAccessCount() const {
        return access_count_;
    }

    /**
     * @brief Gets the last access time.
     *
     * @return The last access time
     */
    std::chrono::steady_clock::time_point getLastAccessTime() const {
        return last_access_time_;
    }

    /**
     * @brief Updates the access statistics.
     */
    void access() {
        ++access_count_;
        last_access_time_ = std::chrono::steady_clock::now();
    }

private:
    Key key_;                                       ///< The key
    Value value_;                                   ///< The value
    size_t size_;                                   ///< The size of the item in bytes
    size_t access_count_;                           ///< The number of times the item has been accessed
    std::chrono::steady_clock::time_point last_access_time_; ///< The last time the item was accessed
};

/**
 * @class CacheLevel
 * @brief Class for a cache level.
 *
 * The CacheLevel class represents a level in the multi-level cache.
 * It provides methods for inserting, retrieving, and evicting items.
 *
 * @tparam Key The key type
 * @tparam Value The value type
 */
template <typename Key, typename Value>
class CacheLevelImpl {
public:
    /**
     * @brief Constructs a new CacheLevelImpl object.
     *
     * @param capacity The capacity of the cache level in bytes
     * @param policy The cache replacement policy
     */
    CacheLevelImpl(size_t capacity, CachePolicy policy)
        : capacity_(capacity), policy_(policy), size_(0), hits_(0), misses_(0), evictions_(0),
          total_lookup_time_(0), total_insert_time_(0), lookup_count_(0), insert_count_(0) {}

    /**
     * @brief Inserts an item into the cache.
     *
     * @param key The key
     * @param value The value
     * @param size The size of the item in bytes
     * @return True if the item was inserted, false otherwise
     */
    bool insert(const Key& key, const Value& value, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Measure insert time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Check if the item is too large for the cache
        if (size > capacity_) {
            return false;
        }
        
        // Check if the item already exists
        auto it = items_.find(key);
        if (it != items_.end()) {
            // Update the item
            size_ -= it->second->getSize();
            size_ += size;
            it->second = std::make_shared<CacheItem<Key, Value>>(key, value, size);
            
            // Update access order based on policy
            updateAccessOrder(key);
            
            // Update insert time statistics
            auto end_time = std::chrono::high_resolution_clock::now();
            total_insert_time_ += std::chrono::duration<double, std::micro>(end_time - start_time).count();
            insert_count_++;
            
            return true;
        }
        
        // Make room for the new item if needed
        while (size_ + size > capacity_ && !access_order_.empty()) {
            evict();
        }
        
        // Check if there's still not enough room
        if (size_ + size > capacity_) {
            return false;
        }
        
        // Insert the new item
        auto item = std::make_shared<CacheItem<Key, Value>>(key, value, size);
        items_[key] = item;
        size_ += size;
        
        // Update access order based on policy
        access_order_.push_back(key);
        key_to_order_[key] = --access_order_.end();
        
        // Update insert time statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        total_insert_time_ += std::chrono::duration<double, std::micro>(end_time - start_time).count();
        insert_count_++;
        
        return true;
    }

    /**
     * @brief Retrieves an item from the cache.
     *
     * @param key The key
     * @param value Reference to store the value
     * @return True if the item was found, false otherwise
     */
    bool get(const Key& key, Value& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Measure lookup time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Check if the item exists
        auto it = items_.find(key);
        if (it == items_.end()) {
            // Item not found
            misses_++;
            
            // Update lookup time statistics
            auto end_time = std::chrono::high_resolution_clock::now();
            total_lookup_time_ += std::chrono::duration<double, std::micro>(end_time - start_time).count();
            lookup_count_++;
            
            return false;
        }
        
        // Item found
        value = it->second->getValue();
        it->second->access();
        hits_++;
        
        // Update access order based on policy
        updateAccessOrder(key);
        
        // Update lookup time statistics
        auto end_time = std::chrono::high_resolution_clock::now();
        total_lookup_time_ += std::chrono::duration<double, std::micro>(end_time - start_time).count();
        lookup_count_++;
        
        return true;
    }

    /**
     * @brief Removes an item from the cache.
     *
     * @param key The key
     * @return True if the item was removed, false otherwise
     */
    bool remove(const Key& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check if the item exists
        auto it = items_.find(key);
        if (it == items_.end()) {
            return false;
        }
        
        // Remove the item
        size_ -= it->second->getSize();
        items_.erase(it);
        
        // Remove from access order
        auto order_it = key_to_order_.find(key);
        if (order_it != key_to_order_.end()) {
            access_order_.erase(order_it->second);
            key_to_order_.erase(order_it);
        }
        
        return true;
    }

    /**
     * @brief Clears the cache.
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        items_.clear();
        access_order_.clear();
        key_to_order_.clear();
        size_ = 0;
    }

    /**
     * @brief Gets the cache statistics.
     *
     * @return The cache statistics
     */
    CacheStats getStats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        CacheStats stats;
        stats.capacity = capacity_;
        stats.size = size_;
        stats.item_count = items_.size();
        stats.hits = hits_;
        stats.misses = misses_;
        stats.evictions = evictions_;
        stats.hit_ratio = (hits_ + misses_ > 0) ? static_cast<double>(hits_) / (hits_ + misses_) : 0.0;
        stats.avg_lookup_time = (lookup_count_ > 0) ? total_lookup_time_ / lookup_count_ : 0.0;
        stats.avg_insert_time = (insert_count_ > 0) ? total_insert_time_ / insert_count_ : 0.0;
        
        return stats;
    }

    /**
     * @brief Gets the capacity of the cache.
     *
     * @return The capacity in bytes
     */
    size_t getCapacity() const {
        return capacity_;
    }

    /**
     * @brief Gets the current size of the cache.
     *
     * @return The size in bytes
     */
    size_t getSize() const {
        return size_;
    }

    /**
     * @brief Gets the number of items in the cache.
     *
     * @return The number of items
     */
    size_t getItemCount() const {
        return items_.size();
    }

    /**
     * @brief Sets the capacity of the cache.
     *
     * @param capacity The capacity in bytes
     */
    void setCapacity(size_t capacity) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        capacity_ = capacity;
        
        // Evict items if needed
        while (size_ > capacity_ && !access_order_.empty()) {
            evict();
        }
    }

    /**
     * @brief Sets the cache replacement policy.
     *
     * @param policy The cache replacement policy
     */
    void setPolicy(CachePolicy policy) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        policy_ = policy;
        
        // Rebuild access order based on the new policy
        rebuildAccessOrder();
    }

private:
    /**
     * @brief Evicts an item from the cache.
     */
    void evict() {
        if (access_order_.empty()) {
            return;
        }
        
        // Get the key to evict based on the policy
        Key key_to_evict;
        
        switch (policy_) {
            case CachePolicy::LRU:
            case CachePolicy::FIFO:
                // Evict the first item in the access order
                key_to_evict = access_order_.front();
                break;
                
            case CachePolicy::LFU: {
                // Evict the item with the lowest access count
                size_t min_access_count = std::numeric_limits<size_t>::max();
                auto min_it = access_order_.end();
                
                for (auto it = access_order_.begin(); it != access_order_.end(); ++it) {
                    auto item_it = items_.find(*it);
                    if (item_it != items_.end() && item_it->second->getAccessCount() < min_access_count) {
                        min_access_count = item_it->second->getAccessCount();
                        min_it = it;
                    }
                }
                
                if (min_it != access_order_.end()) {
                    key_to_evict = *min_it;
                    access_order_.erase(min_it);
                } else {
                    // Fallback to LRU
                    key_to_evict = access_order_.front();
                    access_order_.pop_front();
                }
                break;
            }
                
            case CachePolicy::RANDOM: {
                // Evict a random item
                size_t index = std::rand() % access_order_.size();
                auto it = access_order_.begin();
                std::advance(it, index);
                key_to_evict = *it;
                access_order_.erase(it);
                break;
            }
        }
        
        // Remove the item from the cache
        auto it = items_.find(key_to_evict);
        if (it != items_.end()) {
            size_ -= it->second->getSize();
            items_.erase(it);
            evictions_++;
        }
        
        // Remove from key to order map
        key_to_order_.erase(key_to_evict);
    }

    /**
     * @brief Updates the access order for a key.
     *
     * @param key The key
     */
    void updateAccessOrder(const Key& key) {
        // Update access order based on policy
        auto order_it = key_to_order_.find(key);
        if (order_it != key_to_order_.end()) {
            // Remove from current position
            access_order_.erase(order_it->second);
        }
        
        switch (policy_) {
            case CachePolicy::LRU:
                // Move to the back (most recently used)
                access_order_.push_back(key);
                key_to_order_[key] = --access_order_.end();
                break;
                
            case CachePolicy::LFU:
                // Keep sorted by access count (least frequently used at the front)
                rebuildAccessOrder();
                break;
                
            case CachePolicy::FIFO:
                // Add to the back (will be evicted last)
                access_order_.push_back(key);
                key_to_order_[key] = --access_order_.end();
                break;
                
            case CachePolicy::RANDOM:
                // Add to the back (position doesn't matter for random)
                access_order_.push_back(key);
                key_to_order_[key] = --access_order_.end();
                break;
        }
    }

    /**
     * @brief Rebuilds the access order based on the current policy.
     */
    void rebuildAccessOrder() {
        // Clear current access order
        access_order_.clear();
        key_to_order_.clear();
        
        // Build a vector of keys
        std::vector<Key> keys;
        keys.reserve(items_.size());
        for (const auto& pair : items_) {
            keys.push_back(pair.first);
        }
        
        // Sort keys based on policy
        switch (policy_) {
            case CachePolicy::LRU:
                // Sort by last access time (oldest first)
                std::sort(keys.begin(), keys.end(), [this](const Key& a, const Key& b) {
                    auto a_it = items_.find(a);
                    auto b_it = items_.find(b);
                    if (a_it != items_.end() && b_it != items_.end()) {
                        return a_it->second->getLastAccessTime() < b_it->second->getLastAccessTime();
                    }
                    return false;
                });
                break;
                
            case CachePolicy::LFU:
                // Sort by access count (least frequent first)
                std::sort(keys.begin(), keys.end(), [this](const Key& a, const Key& b) {
                    auto a_it = items_.find(a);
                    auto b_it = items_.find(b);
                    if (a_it != items_.end() && b_it != items_.end()) {
                        return a_it->second->getAccessCount() < b_it->second->getAccessCount();
                    }
                    return false;
                });
                break;
                
            case CachePolicy::FIFO:
                // Keep current order (we don't have insertion time)
                break;
                
            case CachePolicy::RANDOM:
                // Randomize order
                std::random_shuffle(keys.begin(), keys.end());
                break;
        }
        
        // Rebuild access order
        for (const auto& key : keys) {
            access_order_.push_back(key);
            key_to_order_[key] = --access_order_.end();
        }
    }

    size_t capacity_;                                  ///< Cache capacity in bytes
    CachePolicy policy_;                               ///< Cache replacement policy
    size_t size_;                                      ///< Current cache size in bytes
    std::unordered_map<Key, std::shared_ptr<CacheItem<Key, Value>>> items_; ///< Cache items
    std::list<Key> access_order_;                      ///< Access order for replacement policies
    std::unordered_map<Key, typename std::list<Key>::iterator> key_to_order_; ///< Map from keys to access order iterators
    mutable std::mutex mutex_;                         ///< Mutex for thread safety
    
    // Statistics
    size_t hits_;                                      ///< Number of cache hits
    size_t misses_;                                    ///< Number of cache misses
    size_t evictions_;                                 ///< Number of cache evictions
    double total_lookup_time_;                         ///< Total lookup time in microseconds
    double total_insert_time_;                         ///< Total insert time in microseconds
    size_t lookup_count_;                              ///< Number of lookups
    size_t insert_count_;                              ///< Number of inserts
};

/**
 * @class MultiLevelCache
 * @brief Class for a multi-level cache.
 *
 * The MultiLevelCache class provides a multi-level cache system
 * with different levels of caching (L1, L2, L3).
 *
 * @tparam Key The key type
 * @tparam Value The value type
 */
template <typename Key, typename Value>
class MultiLevelCache {
public:
    /**
     * @brief Constructs a new MultiLevelCache object.
     *
     * @param l1_capacity The capacity of the L1 cache in bytes
     * @param l2_capacity The capacity of the L2 cache in bytes
     * @param l3_capacity The capacity of the L3 cache in bytes
     * @param l1_policy The L1 cache replacement policy
     * @param l2_policy The L2 cache replacement policy
     * @param l3_policy The L3 cache replacement policy
     */
    MultiLevelCache(size_t l1_capacity, size_t l2_capacity, size_t l3_capacity,
                   CachePolicy l1_policy = CachePolicy::LRU,
                   CachePolicy l2_policy = CachePolicy::LFU,
                   CachePolicy l3_policy = CachePolicy::FIFO)
        : l1_cache_(l1_capacity, l1_policy),
          l2_cache_(l2_capacity, l2_policy),
          l3_cache_(l3_capacity, l3_policy),
          size_estimator_(nullptr) {}

    /**
     * @brief Inserts an item into the cache.
     *
     * @param key The key
     * @param value The value
     * @param size The size of the item in bytes (optional if size estimator is set)
     * @param level The cache level to insert into (default: ALL)
     * @return True if the item was inserted, false otherwise
     */
    bool insert(const Key& key, const Value& value, size_t size = 0, CacheLevel level = CacheLevel::ALL) {
        // Estimate size if not provided
        if (size == 0 && size_estimator_) {
            size = size_estimator_(value);
        }
        
        // Insert into the specified level(s)
        bool result = true;
        
        if (level == CacheLevel::L1 || level == CacheLevel::ALL) {
            result = result && l1_cache_.insert(key, value, size);
        }
        
        if (level == CacheLevel::L2 || level == CacheLevel::ALL) {
            result = result && l2_cache_.insert(key, value, size);
        }
        
        if (level == CacheLevel::L3 || level == CacheLevel::ALL) {
            result = result && l3_cache_.insert(key, value, size);
        }
        
        return result;
    }

    /**
     * @brief Retrieves an item from the cache.
     *
     * @param key The key
     * @param value Reference to store the value
     * @return True if the item was found, false otherwise
     */
    bool get(const Key& key, Value& value) {
        // Try to get from L1 cache
        if (l1_cache_.get(key, value)) {
            return true;
        }
        
        // Try to get from L2 cache
        if (l2_cache_.get(key, value)) {
            // Promote to L1 cache
            l1_cache_.insert(key, value, size_estimator_ ? size_estimator_(value) : 0);
            return true;
        }
        
        // Try to get from L3 cache
        if (l3_cache_.get(key, value)) {
            // Promote to L2 cache
            l2_cache_.insert(key, value, size_estimator_ ? size_estimator_(value) : 0);
            return true;
        }
        
        return false;
    }

    /**
     * @brief Removes an item from the cache.
     *
     * @param key The key
     * @param level The cache level to remove from (default: ALL)
     * @return True if the item was removed, false otherwise
     */
    bool remove(const Key& key, CacheLevel level = CacheLevel::ALL) {
        bool result = false;
        
        if (level == CacheLevel::L1 || level == CacheLevel::ALL) {
            result = result || l1_cache_.remove(key);
        }
        
        if (level == CacheLevel::L2 || level == CacheLevel::ALL) {
            result = result || l2_cache_.remove(key);
        }
        
        if (level == CacheLevel::L3 || level == CacheLevel::ALL) {
            result = result || l3_cache_.remove(key);
        }
        
        return result;
    }

    /**
     * @brief Clears the cache.
     *
     * @param level The cache level to clear (default: ALL)
     */
    void clear(CacheLevel level = CacheLevel::ALL) {
        if (level == CacheLevel::L1 || level == CacheLevel::ALL) {
            l1_cache_.clear();
        }
        
        if (level == CacheLevel::L2 || level == CacheLevel::ALL) {
            l2_cache_.clear();
        }
        
        if (level == CacheLevel::L3 || level == CacheLevel::ALL) {
            l3_cache_.clear();
        }
    }

    /**
     * @brief Gets the cache statistics.
     *
     * @param level The cache level to get statistics for
     * @return The cache statistics
     */
    CacheStats getStats(CacheLevel level) const {
        switch (level) {
            case CacheLevel::L1:
                return l1_cache_.getStats();
            case CacheLevel::L2:
                return l2_cache_.getStats();
            case CacheLevel::L3:
                return l3_cache_.getStats();
            default:
                // Return combined stats
                CacheStats stats = l1_cache_.getStats();
                CacheStats l2_stats = l2_cache_.getStats();
                CacheStats l3_stats = l3_cache_.getStats();
                
                stats.capacity = stats.capacity + l2_stats.capacity + l3_stats.capacity;
                stats.size = stats.size + l2_stats.size + l3_stats.size;
                stats.item_count = stats.item_count + l2_stats.item_count + l3_stats.item_count;
                stats.hits = stats.hits + l2_stats.hits + l3_stats.hits;
                stats.misses = stats.misses;  // Only count L3 misses as overall misses
                stats.evictions = stats.evictions + l2_stats.evictions + l3_stats.evictions;
                stats.hit_ratio = (stats.hits + stats.misses > 0) ? 
                    static_cast<double>(stats.hits) / (stats.hits + stats.misses) : 0.0;
                stats.avg_lookup_time = (stats.avg_lookup_time + l2_stats.avg_lookup_time + l3_stats.avg_lookup_time) / 3.0;
                stats.avg_insert_time = (stats.avg_insert_time + l2_stats.avg_insert_time + l3_stats.avg_insert_time) / 3.0;
                
                return stats;
        }
    }

    /**
     * @brief Sets the capacity of a cache level.
     *
     * @param level The cache level
     * @param capacity The capacity in bytes
     */
    void setCapacity(CacheLevel level, size_t capacity) {
        switch (level) {
            case CacheLevel::L1:
                l1_cache_.setCapacity(capacity);
                break;
            case CacheLevel::L2:
                l2_cache_.setCapacity(capacity);
                break;
            case CacheLevel::L3:
                l3_cache_.setCapacity(capacity);
                break;
            default:
                // Set all levels
                l1_cache_.setCapacity(capacity / 4);  // 25% for L1
                l2_cache_.setCapacity(capacity / 4);  // 25% for L2
                l3_cache_.setCapacity(capacity / 2);  // 50% for L3
                break;
        }
    }

    /**
     * @brief Sets the cache replacement policy for a cache level.
     *
     * @param level The cache level
     * @param policy The cache replacement policy
     */
    void setPolicy(CacheLevel level, CachePolicy policy) {
        switch (level) {
            case CacheLevel::L1:
                l1_cache_.setPolicy(policy);
                break;
            case CacheLevel::L2:
                l2_cache_.setPolicy(policy);
                break;
            case CacheLevel::L3:
                l3_cache_.setPolicy(policy);
                break;
            default:
                // Set all levels
                l1_cache_.setPolicy(policy);
                l2_cache_.setPolicy(policy);
                l3_cache_.setPolicy(policy);
                break;
        }
    }

    /**
     * @brief Sets the size estimator function.
     *
     * @param estimator The size estimator function
     */
    void setSizeEstimator(std::function<size_t(const Value&)> estimator) {
        size_estimator_ = estimator;
    }

private:
    CacheLevelImpl<Key, Value> l1_cache_;  ///< L1 cache (fastest, smallest)
    CacheLevelImpl<Key, Value> l2_cache_;  ///< L2 cache
    CacheLevelImpl<Key, Value> l3_cache_;  ///< L3 cache (slowest, largest)
    std::function<size_t(const Value&)> size_estimator_;  ///< Function to estimate the size of a value
};

} // namespace Cache
