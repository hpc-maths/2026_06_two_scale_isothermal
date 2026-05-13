// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Giuseppe Orlando, 2026
//
#pragma once

#include <samurai/schemes/fv.hpp>

#include "../barotropic_eos.hpp"
#include "../utilities.hpp"

namespace samurai {
  using namespace EquationData;

  /**
   * Generic class to compute the flux between a left and right state
   */
  template<class Field>
  class Flux {
  public:
    // Definitions and sanity checks
    static_assert(Field::dim == EquationData::dim, "The spatial dimensions between Field and the parameter list do not match");
    static_assert(Field::n_comp == EquationData::NVARS, "The number of elements in the state does not correspond to the number of equations");
    static constexpr std::size_t stencil_size = 4;

    using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, Field, Field>;

    template<class Field_Vect>
    using cfg_st = FluxConfig<SchemeType::NonLinear, stencil_size, Field, Field_Vect>;

    using Number = typename Field::value_type; // Shortcut for the arithmetic type

    /**
     * Class constructor
     * @param EOS_phase_liq_ liquid equation of state
     * @param EOS_phase_gas_ gas equation of state
     * @param sigma_ surface tension coefficient
     */
    Flux(const LinearizedBarotropicEOS<Number>& EOS_phase_liq_,
         const LinearizedBarotropicEOS<Number>& EOS_phase_gas_,
         const Number sigma_);

  protected:
    const LinearizedBarotropicEOS<Number>& EOS_phase_liq;
    const LinearizedBarotropicEOS<Number>& EOS_phase_gas;

    const Number sigma; /*!< Surface tension parameter */

    /**
     * Evaluate the 'continuous' flux
     * @param q state
     * @param curr_d current direction
     * @param grad_alpha_l gradient of large-scale volume fraction (needed for capillarity)
     */
    FluxValue<cfg> evaluate_continuous_flux(const FluxValue<cfg>& q,
                                            const std::size_t curr_d,
                                            const auto& grad_alpha_l);

    /**
     * Evaluate the hyperbolic operator
     * @param q state
     * @param curr_d current direction
     */
    FluxValue<cfg> evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                                const std::size_t curr_d);

    /**
     * Evaluate the surface tension operator
     * @param grad_alpha_l gradient of large-scale volume fraction
     * @param curr_d current direction
     */
    template<class Field_Vect>
    FluxValue<cfg_st<Field_Vect>> evaluate_surface_tension_operator(const auto& grad_alpha_l,
                                                                    const std::size_t curr_d);

    /**
     * Conversion from conserved to primitive variables
     * @param cons conserved variables
     * @return prim primitive variables
     */
    FluxValue<cfg> cons2prim(const FluxValue<cfg>& cons) const;

    /**
     * Conversion from primitive to conserved variables
     * @param prim primitive variables
     * @return cons conserved variables
     */
    FluxValue<cfg> prim2cons(const FluxValue<cfg>& prim) const;
  };

  // Class constructor in order to be able to work with the equation of state
  //
  template<class Field>
  Flux<Field>::Flux(const LinearizedBarotropicEOS<Number>& EOS_phase_liq_,
                    const LinearizedBarotropicEOS<Number>& EOS_phase_gas_,
                    const Number sigma_):
    EOS_phase_liq(EOS_phase_liq_), EOS_phase_gas(EOS_phase_gas_), sigma(sigma_) {}

  // Evaluate the 'continuous flux'
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  Flux<Field>::evaluate_continuous_flux(const FluxValue<cfg>& q,
                                        const std::size_t curr_d,
                                        const auto& grad_alpha_l) {
    // Initialize the resulting variable with the hyperbolic operator
    FluxValue<cfg> res = this->evaluate_hyperbolic_operator(q, curr_d);

    // Add the contribution due to surface tension
    res += this->evaluate_surface_tension_operator(grad_alpha_l, curr_d);

    return res;
  }

  // Evaluate the hyperbolic part of the 'continuous' flux
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  Flux<Field>::evaluate_hyperbolic_operator(const FluxValue<cfg>& q,
                                            const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < Field::dim);

    // Initialize the resulting variable
    FluxValue<cfg> res = q;

    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto m_l = q(Ml_INDEX);
    const auto m_g = q(Mg_INDEX);
    const auto m_d = q(Md_INDEX);

    // Compute the current velocity
    const auto m_liq   = m_l + m_d;
    const auto rho     = m_liq + m_g;
    const auto inv_rho = static_cast<Number>(1.0)/rho;
    const auto vel_d   = q(RHO_U_INDEX + curr_d)*inv_rho;

    // Multiply the state the velocity along the direction of interest
    res(Ml_INDEX) *= vel_d;
    res(Mg_INDEX) *= vel_d;
    res(Md_INDEX) *= vel_d;
    res(RHO_Z_INDEX) *= vel_d;
    res(RHO_ALPHA_l_INDEX) *= vel_d;
    for(std::size_t d = 0; d < Field::dim; ++d) {
      res(RHO_U_INDEX + d) *= vel_d;
    }

    // Compute and add the contribution due to the pressure
    const auto alpha_l   = q(RHO_ALPHA_l_INDEX)*inv_rho;
    const auto alpha_d   = alpha_l*m_d/m_l; // TODO: Add a check in case of zero volume fraction
    const auto alpha_liq = alpha_l + alpha_d;
    const auto rho_liq   = m_liq/alpha_liq; // TODO: Add a check in case of zero volume fraction
    /*NOTE: Relation alpha_l/Y_l = (alpha_l + alpha_d)/(Y_l + Y_d) holds!!! */
    const auto p_liq     = EOS_phase_liq.pres_value(rho_liq);

    const auto alpha_g = static_cast<Number>(1.0) - alpha_liq;
    const auto rho_g   = m_g/alpha_g; // TODO: Add a check in case of zero volume fraction
    const auto p_g     = EOS_phase_gas.pres_value(rho_g);

    const auto Sigma_d = q(RHO_Z_INDEX)/std::cbrt(rho_liq*rho_liq);

    const auto p       = alpha_liq*p_liq
                       + alpha_g*p_g
                       - static_cast<Number>(2.0/3.0)*sigma*Sigma_d;

    res(RHO_U_INDEX + curr_d) += p;

    return res;
  }

  // Evaluate the surface tension operator
  //
  template<class Field>
  template<class Field_Vect>
  FluxValue<typename Flux<Field>::template cfg_st<Field_Vect>>
  Flux<Field>::evaluate_surface_tension_operator(const auto& grad_alpha_l,
                                                 const std::size_t curr_d) {
    // Sanity check in terms of dimensions
    assert(curr_d < Field::dim);

    // Initialize the resulting variable
    FluxValue<cfg_st<Field_Vect>> res;

    // Set to zero all the contributions
    res.fill(static_cast<Number>(0.0));

    // Add the contribution due to surface tension
    auto mod2_grad_alpha_l = static_cast<Number>(0.0);
    for(std::size_t d = 0; d < Field::dim; ++d) {
      mod2_grad_alpha_l += grad_alpha_l[d]*grad_alpha_l[d];
    }
    const auto mod_grad_alpha_l = std::sqrt(mod2_grad_alpha_l);

    const auto n  = grad_alpha_l/(mod_grad_alpha_l + static_cast<Number>(1e-10));
    const auto nx = n(0);
    const auto ny = n(1);

    if(curr_d == 0) {
      res(RHO_U_INDEX) += sigma*(nx*nx - static_cast<Number>(1.0))*mod_grad_alpha_l;
      res(RHO_U_INDEX + 1) += sigma*nx*ny*mod_grad_alpha_l;
    }
    else if(curr_d == 1) {
      res(RHO_U_INDEX) += sigma*nx*ny*mod_grad_alpha_l;
      res(RHO_U_INDEX + 1) += sigma*(ny*ny - static_cast<Number>(1.0))*mod_grad_alpha_l;
    }

    return res;
  }

  // Conversion from conserved to primitive variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  Flux<Field>::cons2prim(const FluxValue<cfg>& cons) const {
    FluxValue<cfg> prim;

    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto m_l = cons(Ml_INDEX);
    const auto m_g = cons(Mg_INDEX);
    const auto m_d = cons(Md_INDEX);

    // Compute primitive variables
    const auto m_liq   = m_l + m_d;
    const auto rho     = m_liq + m_g;
    const auto inv_rho = static_cast<Number>(1.0)/rho;

    const auto alpha_l  = cons(RHO_ALPHA_l_INDEX)*inv_rho;
    prim(ALPHA_l_INDEX) = alpha_l;

    const auto alpha_d   = alpha_l*m_d/m_l;
    prim(ALPHA_2d_INDEX) = alpha_d/(static_cast<Number>(1.0) - alpha_l);

    const auto alpha_liq = alpha_l + alpha_d;
    const auto rho_liq   = m_liq/alpha_liq; // TODO: Add a check in case of zero volume fraction
    prim(Pl_INDEX)       = EOS_phase_liq.pres_value(rho_liq);

    const auto rho_g = m_g/(static_cast<Number>(1.0) - alpha_liq);
                       // TODO: Add a check in case of zero volume fraction
    prim(Pg_INDEX)   = EOS_phase_gas.pres_value(rho_g);

    for(std::size_t d = 0; d < Field::dim; ++d) {
      prim(U_INDEX + d) = cons(RHO_U_INDEX + d)*inv_rho;
    }

    prim(Z_INDEX) = cons(RHO_Z_INDEX)*inv_rho;

    return prim;
  }

  // Conversion from primitive to conserved variables
  //
  template<class Field>
  FluxValue<typename Flux<Field>::cfg>
  Flux<Field>::prim2cons(const FluxValue<cfg>& prim) const {
    FluxValue<cfg> cons;

    // Pre-fetch some variables used multiple times in order to exploit possible vectorization
    const auto alpha_l = prim(ALPHA_l_INDEX);
    const auto alpha_d = prim(ALPHA_2d_INDEX)*(static_cast<Number>(1.0) - alpha_l);

    // Compute conserved variables
    const auto rho_liq = EOS_phase_liq.rho_value(prim(Pl_INDEX));

    const auto m_l = alpha_l*rho_liq;
    cons(Ml_INDEX) = m_l;

    const auto m_d = alpha_d*rho_liq;
    cons(Md_INDEX) = m_d;

    const auto rho_g = EOS_phase_gas.rho_value(prim(Pg_INDEX));
    const auto m_g   = (static_cast<Number>(1.0) - alpha_l - alpha_d)*rho_g;
    cons(Mg_INDEX)   = m_g;

    const auto rho          = m_l + m_g + m_d;
    cons(RHO_ALPHA_l_INDEX) = rho*alpha_l;

    for(std::size_t d = 0; d < Field::dim; ++d) {
      cons(RHO_U_INDEX + d) = rho*prim(U_INDEX + d);
    }

    cons(RHO_Z_INDEX) = rho*prim(Z_INDEX);

    return cons;
  }

} // end namespace samurai
