#pragma once
/**
 * @file recovery_manager.h
 * @brief Defines the RecoveryManager class for graceful degradation.
 *
 * This file contains the declaration of the RecoveryManager class, which provides
 * mechanisms for graceful degradation and recovery from errors in QDSim.
 *
 * Author: Dr. Mazharuddin Mohammed
 */

#include "error_handling.h"
#include <functional>
#include <map>
#include <vector>
#include <memory>
#include <string>
#include <iostream>

/**
 * @namespace Recovery
 * @brief Namespace for recovery utilities.
 */
namespace Recovery {

/**
 * @enum RecoveryStrategy
 * @brief Enumeration of recovery strategies.
 */
enum class RecoveryStrategy {
    RETRY,              ///< Retry the operation
    USE_DEFAULT_VALUE,  ///< Use a default value
    USE_PREVIOUS_VALUE, ///< Use the previous value
    SKIP_OPERATION,     ///< Skip the operation
    FALLBACK_ALGORITHM, ///< Use a fallback algorithm
    ABORT_OPERATION     ///< Abort the operation
};

/**
 * @class RecoveryManager
 * @brief Manager for graceful degradation and recovery from errors.
 *
 * This class provides mechanisms for graceful degradation and recovery from errors
 * in QDSim. It allows registering recovery strategies for different error codes,
 * and executing those strategies when errors occur.
 */
class RecoveryManager {
public:
    /**
     * @brief Gets the singleton instance of the RecoveryManager.
     *
     * @return The singleton instance
     */
    static RecoveryManager& instance() {
        static RecoveryManager instance;
        return instance;
    }
    
    /**
     * @brief Registers a recovery strategy for an error code.
     *
     * @param code The error code
     * @param strategy The recovery strategy
     * @param handler The handler function to execute for the strategy
     */
    void register_strategy(ErrorHandling::ErrorCode code, RecoveryStrategy strategy,
                          std::function<void(const ErrorHandling::QDSimException&)> handler) {
        strategies_[code][strategy] = handler;
    }
    
    /**
     * @brief Executes a recovery strategy for an error.
     *
     * @param e The exception that occurred
     * @param strategy The recovery strategy to execute
     * @return True if a recovery strategy was executed, false otherwise
     */
    bool recover(const ErrorHandling::QDSimException& e, RecoveryStrategy strategy) {
        auto it_code = strategies_.find(e.code());
        if (it_code != strategies_.end()) {
            auto it_strategy = it_code->second.find(strategy);
            if (it_strategy != it_code->second.end()) {
                // Execute the recovery strategy
                it_strategy->second(e);
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief Executes all registered recovery strategies for an error.
     *
     * @param e The exception that occurred
     * @return True if at least one recovery strategy was executed, false otherwise
     */
    bool recover_all(const ErrorHandling::QDSimException& e) {
        auto it_code = strategies_.find(e.code());
        if (it_code != strategies_.end()) {
            bool recovered = false;
            for (const auto& strategy : it_code->second) {
                // Execute the recovery strategy
                strategy.second(e);
                recovered = true;
            }
            return recovered;
        }
        return false;
    }
    
    /**
     * @brief Executes a recovery strategy for an error code.
     *
     * @param code The error code
     * @param strategy The recovery strategy to execute
     * @param message The error message
     * @return True if a recovery strategy was executed, false otherwise
     */
    bool recover(ErrorHandling::ErrorCode code, RecoveryStrategy strategy, const std::string& message = "") {
        ErrorHandling::QDSimException e(code, message);
        return recover(e, strategy);
    }
    
    /**
     * @brief Executes a function with recovery.
     *
     * This method executes a function and, if an exception is thrown, attempts to
     * recover using the specified strategy. If recovery is successful, the function
     * is executed again.
     *
     * @param func The function to execute
     * @param strategy The recovery strategy to use if an exception is thrown
     * @param max_retries The maximum number of retries
     * @return True if the function was executed successfully, false otherwise
     */
    template <typename Func>
    bool execute_with_recovery(Func func, RecoveryStrategy strategy, int max_retries = 3) {
        int retries = 0;
        while (retries <= max_retries) {
            try {
                func();
                return true;
            } catch (const ErrorHandling::QDSimException& e) {
                QDSIM_LOG_WARNING("Error occurred: " + std::string(e.what()));
                
                // Attempt to recover
                bool recovered = recover(e, strategy);
                if (!recovered) {
                    QDSIM_LOG_ERROR("Failed to recover from error: " + std::string(e.what()));
                    return false;
                }
                
                // Increment retry counter
                ++retries;
                
                // Log retry
                QDSIM_LOG_INFO("Retrying operation (attempt " + std::to_string(retries) + " of " + std::to_string(max_retries) + ")");
            } catch (const std::exception& e) {
                QDSIM_LOG_ERROR("Unhandled exception: " + std::string(e.what()));
                return false;
            }
        }
        
        QDSIM_LOG_ERROR("Maximum number of retries reached");
        return false;
    }
    
    /**
     * @brief Provides a default value for a type.
     *
     * @tparam T The type of the default value
     * @return The default value
     */
    template <typename T>
    static T default_value() {
        return T();
    }
    
    /**
     * @brief Provides a fallback algorithm for a specific operation.
     *
     * @tparam Func The type of the fallback function
     * @param operation_name The name of the operation
     * @param fallback The fallback function
     */
    template <typename Func>
    void register_fallback_algorithm(const std::string& operation_name, Func fallback) {
        fallback_algorithms_[operation_name] = [fallback](const void* args) {
            return fallback();
        };
    }
    
    /**
     * @brief Executes a fallback algorithm for a specific operation.
     *
     * @param operation_name The name of the operation
     * @param args The arguments for the fallback algorithm
     * @return True if a fallback algorithm was executed, false otherwise
     */
    bool execute_fallback_algorithm(const std::string& operation_name, const void* args = nullptr) {
        auto it = fallback_algorithms_.find(operation_name);
        if (it != fallback_algorithms_.end()) {
            it->second(args);
            return true;
        }
        return false;
    }
    
private:
    /**
     * @brief Constructs a new RecoveryManager object.
     */
    RecoveryManager() {
        // Register default recovery strategies
        register_default_strategies();
    }
    
    /**
     * @brief Registers default recovery strategies.
     */
    void register_default_strategies() {
        // Register default recovery strategies for common error codes
        
        // Matrix not positive definite
        register_strategy(ErrorHandling::ErrorCode::MATRIX_NOT_POSITIVE_DEFINITE, RecoveryStrategy::FALLBACK_ALGORITHM,
                         [](const ErrorHandling::QDSimException& e) {
                             QDSIM_LOG_INFO("Using regularization to make matrix positive definite");
                             // The actual regularization is implemented in the solver
                         });
        
        // Solver convergence failed
        register_strategy(ErrorHandling::ErrorCode::SOLVER_CONVERGENCE_FAILED, RecoveryStrategy::RETRY,
                         [](const ErrorHandling::QDSimException& e) {
                             QDSIM_LOG_INFO("Retrying solver with increased damping");
                             // The actual damping increase is implemented in the solver
                         });
        
        // Self-consistent divergence
        register_strategy(ErrorHandling::ErrorCode::SELF_CONSISTENT_DIVERGENCE, RecoveryStrategy::FALLBACK_ALGORITHM,
                         [](const ErrorHandling::QDSimException& e) {
                             QDSIM_LOG_INFO("Using simplified model for self-consistent solution");
                             // The actual simplified model is implemented in the solver
                         });
        
        // Memory allocation failed
        register_strategy(ErrorHandling::ErrorCode::MEMORY_ALLOCATION_FAILED, RecoveryStrategy::FALLBACK_ALGORITHM,
                         [](const ErrorHandling::QDSimException& e) {
                             QDSIM_LOG_INFO("Using out-of-core algorithm to reduce memory usage");
                             // The actual out-of-core algorithm is implemented in the solver
                         });
    }
    
    // Map of error codes to recovery strategies
    std::map<ErrorHandling::ErrorCode, std::map<RecoveryStrategy, std::function<void(const ErrorHandling::QDSimException&)>>> strategies_;
    
    // Map of operation names to fallback algorithms
    std::map<std::string, std::function<void(const void*)>> fallback_algorithms_;
};

/**
 * @class ScopedRecovery
 * @brief RAII class for executing code with recovery.
 *
 * This class provides a RAII mechanism for executing code with recovery.
 * It ensures that resources are properly cleaned up even if an exception is thrown.
 */
class ScopedRecovery {
public:
    /**
     * @brief Constructs a new ScopedRecovery object.
     *
     * @param cleanup The cleanup function to execute
     */
    ScopedRecovery(std::function<void()> cleanup) : cleanup_(cleanup) {}
    
    /**
     * @brief Destructor for the ScopedRecovery object.
     *
     * Executes the cleanup function.
     */
    ~ScopedRecovery() {
        try {
            cleanup_();
        } catch (const std::exception& e) {
            QDSIM_LOG_ERROR("Error in cleanup: " + std::string(e.what()));
        }
    }
    
private:
    std::function<void()> cleanup_; ///< The cleanup function
};

} // namespace Recovery

// Convenience macros for recovery
#define QDSIM_RECOVER(e, strategy) \
    Recovery::RecoveryManager::instance().recover(e, strategy)

#define QDSIM_RECOVER_ALL(e) \
    Recovery::RecoveryManager::instance().recover_all(e)

#define QDSIM_EXECUTE_WITH_RECOVERY(func, strategy, max_retries) \
    Recovery::RecoveryManager::instance().execute_with_recovery(func, strategy, max_retries)

#define QDSIM_DEFAULT_VALUE(T) \
    Recovery::RecoveryManager::default_value<T>()

#define QDSIM_REGISTER_FALLBACK(operation_name, fallback) \
    Recovery::RecoveryManager::instance().register_fallback_algorithm(operation_name, fallback)

#define QDSIM_EXECUTE_FALLBACK(operation_name, args) \
    Recovery::RecoveryManager::instance().execute_fallback_algorithm(operation_name, args)

#define QDSIM_SCOPED_RECOVERY(cleanup) \
    Recovery::ScopedRecovery scoped_recovery_##__LINE__(cleanup)
