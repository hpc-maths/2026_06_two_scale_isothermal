// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <samurai/schemes/fv.hpp>

/**
 * Useful parameters and enumerators
 */
namespace EquationData {
  // Declare spatial dimension
  static constexpr std::size_t dim = 2;

  // Use auxiliary variables for the indices for the sake of generality
  static constexpr std::size_t Ml_INDEX          = 0;
  static constexpr std::size_t Mg_INDEX          = 1;
  static constexpr std::size_t Md_INDEX          = 2;
  static constexpr std::size_t RHO_Z_INDEX       = 3;
  static constexpr std::size_t RHO_ALPHA_l_INDEX = 4;
  static constexpr std::size_t RHO_U_INDEX       = 5;

  // Save also the total number of (scalar) variables
  static constexpr std::size_t NVARS = 5 + dim;

  // Use auxiliary variables for the indices also for primitive variables for the sake of generality
  static constexpr std::size_t ALPHA_l_INDEX  = RHO_ALPHA_l_INDEX;
  static constexpr std::size_t U_INDEX        = RHO_U_INDEX;
  static constexpr std::size_t Z_INDEX        = RHO_Z_INDEX;
  static constexpr std::size_t Pl_INDEX       = Ml_INDEX;
  static constexpr std::size_t Pg_INDEX       = Mg_INDEX;
  static constexpr std::size_t ALPHA_2d_INDEX = Md_INDEX;
}

/**
 * Useful auxiliary functions not related to a specific class or instance
 */
namespace Utilities {
  /**
   * Auxiliary function to convert unsigned to string
   * @param value the value to be converted
   * @param digits number of digits desidered in the conversion (5 by default)
   */
  template<typename T>
  std::string unsigned_to_string(const T value, const unsigned digits = 5) {
    std::string lc_string = std::to_string(value);

    if(lc_string.size() < digits) {
      // We have to add the padding zeros in front of the number
      const unsigned int padding_position = (lc_string[0] == '-') ? 1 : 0;

      const std::string padding(digits - lc_string.size(), '0');
      lc_string.insert(padding_position, padding);
    }

    return lc_string;
  }

  /**
   * Auxiliary function for sum with mpi
   * @param local_val the local value to be summed
   */
  template<typename T>
  inline T mpi_reduce_sum(const T local_val) {
    #ifdef SAMURAI_WITH_MPI
      double result;
      MPI_Allreduce(&local_val, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      return static_cast<T>(result);
    #else
      return local_val;
    #endif
  }

  /**
   * Auxiliary function for max with mpi
   * @param local_val the local value for which we llok for the max
   */
  template<typename T>
  inline T mpi_reduce_max(const T local_val) {
    #ifdef SAMURAI_WITH_MPI
      double result;
      MPI_Allreduce(&local_val, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      return static_cast<T>(result);
    #else
      return local_val;
    #endif
  }

  /**
   * Auxiliary function to write data (time series)
   * @param f stream where to write data
   * @param time current time instant
   * @param value value to be written
   * @param prec precision to be employed in output (12 digits by default)
   */
  template<typename T>
  inline void write_data(std::ofstream& f,
                         const T time, const T value,
                         const int prec = 12) {
    f << std::fixed << std::setprecision(prec)
      << time << '\t'
      << value
      << std::endl;
  }

  /**
   * Reconstruction for second order scheme (let's say for the face i+1/2)
   * @param primLL primitive variables at i-1
   * @param primL primitive variables at i
   * @param primR primitive variables at i+1
   * @param primRR primitive variables at i+2
   * @param primL_recon reconstructed primitive variables at i
   * @param primR_recon reconstructed primitive variables at i+1
   */
  template<class Field>
  void perform_reconstruction(const auto& primLL,
                              const auto& primL,
                              const auto& primR,
                              const auto& primRR,
                              auto& primL_recon,
                              auto& primR_recon) {
    using Number = typename Field::value_type; // Define the shortcut for the arithmetic type

    // Initialize with the original state
    primL_recon = primL;
    primR_recon = primR;

    // Perform the reconstruction
    const auto beta = static_cast<Number>(1.0); // MINMOD limiter
    for(std::size_t comp = 0; comp < Field::n_comp; ++comp) {
      if(primR(comp) - primL(comp) > static_cast<Number>(0.0)) {
        primL_recon(comp) += static_cast<Number>(0.5)*
                             std::max(static_cast<Number>(0.0),
                                      std::max(std::min(beta*(primL(comp) - primLL(comp)),
                                                        primR(comp) - primL(comp)),
                                               std::min(primL(comp) - primLL(comp),
                                                        beta*(primR(comp) - primL(comp)))));
      }
      else if(primR(comp) - primL(comp) < static_cast<Number>(0.0)) {
        primL_recon(comp) += static_cast<Number>(0.5)*
                             std::min(static_cast<Number>(0.0),
                                      std::min(std::max(beta*(primL(comp) - primLL(comp)),
                                                        primR(comp) - primL(comp)),
                                               std::max(primL(comp) - primLL(comp),
                                                        beta*(primR(comp) - primL(comp)))));
      }

      if(primRR(comp) - primR(comp) > static_cast<Number>(0.0)) {
        primR_recon(comp) -= static_cast<Number>(0.5)*
                             std::max(static_cast<Number>(0.0),
                                      std::max(std::min(beta*(primR(comp) - primL(comp)),
                                                        primRR(comp) - primR(comp)),
                                               std::min(primR(comp) - primL(comp),
                                                        beta*(primRR(comp) - primR(comp)))));
      }
      else if(primRR(comp) - primR(comp) < static_cast<Number>(0.0)) {
        primR_recon(comp) -= static_cast<Number>(0.5)*
                             std::min(static_cast<Number>(0.0),
                                      std::min(std::max(beta*(primR(comp) - primL(comp)),
                                                        primRR(comp) - primR(comp)),
                                               std::max(primR(comp) - primL(comp),
                                                        beta*(primRR(comp) - primR(comp)))));
      }
    }
  }
} // end of namespace
