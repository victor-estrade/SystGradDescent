from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from neyman.inferences import batch_hessian
from ast import literal_eval

import json
import itertools as it
import tensorflow_probability as tfp

from collections import OrderedDict
from fisher_matrix import FisherMatrix
from template_model import int_quad_lin

import config

ds = tfp.distributions

class HiggsTemplateModel(object):

  def __init__(self, multiple_pars=False):

    self.multiple_pars = multiple_pars
    if multiple_pars:
      shape_pars = (None,)
    else:
      shape_pars = ()

    def default_ph_value(val):
      if multiple_pars:
        return [val, ]
      else:
        return val

    self.tau_energy_init = tf.placeholder_with_default(
        config.CALIBRATED_TAU_ENERGY_SCALE,
        shape=(), name="tau_energy_init")
    self.tau_energy_shift = tf.placeholder_with_default(
        config.CALIBRATED_TAU_ENERGY_SCALE_ERROR,
        shape=(), name="tau_energy_shift")

    self.tau_energy = tf.placeholder_with_default(
      default_ph_value(config.CALIBRATED_TAU_ENERGY_SCALE), shape=shape_pars, name="tau_energy")

    self.jet_energy_init = tf.placeholder_with_default(
        config.CALIBRATED_JET_ENERGY_SCALE, shape=(), name="jet_energy_init")
    self.jet_energy_shift = tf.placeholder_with_default(
        config.CALIBRATED_JET_ENERGY_SCALE_ERROR, shape=(), name="jet_energy_shift")

    self.jet_energy = tf.placeholder_with_default(
      default_ph_value(config.CALIBRATED_JET_ENERGY_SCALE), shape=shape_pars, name="jet_energy")

    self.lep_energy_init = tf.placeholder_with_default(
        config.CALIBRATED_LEP_ENERGY_SCALE, shape=(), name="lep_energy_init")
    self.lep_energy_shift = tf.placeholder_with_default(
        config.CALIBRATED_LEP_ENERGY_SCALE_ERROR, shape=(), name="lep_energy_shift")

    self.lep_energy = tf.placeholder_with_default(
      default_ph_value(config.CALIBRATED_LEP_ENERGY_SCALE), shape=shape_pars, name="lep_energy")

    self.soft_term_init = tf.placeholder_with_default(
        config.CALIBRATED_SIGMA_SOFT, shape=(), name="soft_term_init")
    self.soft_term_shift = tf.placeholder_with_default(
        config.CALIBRATED_SIGMA_SOFT_ERROR, shape=(), name="soft_term_shift")

    self.soft_term = tf.placeholder_with_default(
      default_ph_value(config.CALIBRATED_SIGMA_SOFT), shape=shape_pars, name="soft_term")

    self.nasty_bkg_init = tf.placeholder_with_default(
        config.CALIBRATED_NASTY_BKG, shape=(), name="nasty_bkg_init")
    self.nasty_bkg_shift = tf.placeholder_with_default(
        config.CALIBRATED_NASTY_BKG, shape=(), name="nasty_bkg_shift")

    self.nasty_bkg = tf.placeholder_with_default(
      default_ph_value(config.CALIBRATED_NASTY_BKG), shape=shape_pars, name="nasty_bkg")




    # background template
    self.b_nom = tf.placeholder(
        dtype=tf.float32, shape=(None,), name="b_nom")
    self.b_up = tf.placeholder(
        dtype=tf.float32, shape=(None, None), name="b_up")
    self.b_dw = tf.placeholder(
        dtype=tf.float32, shape=(None, None), name="b_dw")
    
    # signal template
    self.s_nom = tf.placeholder(
        dtype=tf.float32, shape=(None,), name="s_nom")
    self.s_up = tf.placeholder(
        dtype=tf.float32, shape=(None, None), name="s_up")
    self.s_dw = tf.placeholder(
        dtype=tf.float32, shape=(None, None), name="s_dw")    

    self.alpha_pars = [[
                        (self.tau_energy - self.tau_energy_init)
                        / self.tau_energy_shift,
                        (self.jet_energy - self.jet_energy_init)
                        / self.jet_energy_shift,
                        (self.lep_energy - self.lep_energy_init)
                        / self.lep_energy_shift,
                        (self.soft_term - self.soft_term_init)
                        / self.soft_term_shift,
                        (self.nasty_bkg - self.nasty_bkg_init)
                        / self.nasty_bkg_shift,
                        ]]

    # bkg_shape shape is (n_par_inst, n_bins, 1) if multiple_pars
    self.bkg_shape = int_quad_lin(self.alpha_pars,
                                  self.b_nom, self.b_up, self.b_dw,
                                  multiple_pars=multiple_pars)[0]
    # sig shape shape is (n_par_inst, n_bins, 1) if multiple_pars
    self.sig_shape = int_quad_lin(self.alpha_pars,
                                  self.s_nom, self.s_up, self.s_dw,
                                  multiple_pars=multiple_pars)[0]
    # expected amount of signal
    self.mu = tf.placeholder_with_default(default_ph_value(1.),
                                             shape=shape_pars, name="mu")

    if multiple_pars:
      mu = tf.expand_dims(self.mu, axis=-1, name="expanded_mu")
    else:
      mu = self.mu

    self.t_exp = tf.cast(mu * self.sig_shape +
                         self.bkg_shape,
                         dtype=tf.float64, name="t_exp")

    # placeholder for observed data
    self.obs = tf.placeholder(dtype=tf.float64, shape=(None,), name="obs")

    self.h_pois = ds.Poisson(self.t_exp)
    self.h_nll = - \
        tf.cast(tf.reduce_sum(self.h_pois.log_prob(self.obs), axis=-1),
                dtype=tf.float32)

    self.all_pars = OrderedDict([('mu', self.mu),
                                 ('tau_energy', self.tau_energy),
                                 ('jet_energy', self.jet_energy),
                                 ('lep_energy', self.lep_energy),
                                 ('soft_term', self.soft_term),
                                 ('nasty_bkg', self.nasty_bkg),
                                 ])

    pars = list(self.all_pars.values())

    self.h_hess, self.h_grad = batch_hessian(self.h_nll, pars)

  def templates_from_dict(self, templates,
                          tau_energy=[
                                      config.CALIBRATED_TAU_ENERGY_SCALE,
                                      config.CALIBRATED_TAU_ENERGY_SCALE+config.CALIBRATED_TAU_ENERGY_SCALE_ERROR,
                                      config.CALIBRATED_TAU_ENERGY_SCALE-config.CALIBRATED_TAU_ENERGY_SCALE_ERROR,
                                      ],
                          lep_energy=[
                                      config.CALIBRATED_LEP_ENERGY_SCALE,
                                      config.CALIBRATED_LEP_ENERGY_SCALE+config.CALIBRATED_LEP_ENERGY_SCALE_ERROR,
                                      config.CALIBRATED_LEP_ENERGY_SCALE-config.CALIBRATED_LEP_ENERGY_SCALE_ERROR,
                                      ],
                          jet_energy=[
                                      config.CALIBRATED_JET_ENERGY_SCALE,
                                      config.CALIBRATED_JET_ENERGY_SCALE+config.CALIBRATED_JET_ENERGY_SCALE_ERROR,
                                      config.CALIBRATED_JET_ENERGY_SCALE-config.CALIBRATED_JET_ENERGY_SCALE_ERROR,
                                      ],
                          soft_term=[
                                      config.CALIBRATED_SIGMA_SOFT,
                                      config.CALIBRATED_SIGMA_SOFT+config.CALIBRATED_SIGMA_SOFT_ERROR,
                                      config.CALIBRATED_SIGMA_SOFT-config.CALIBRATED_SIGMA_SOFT_ERROR,
                                      ],
                          nasty_bkg=[
                                      config.CALIBRATED_NASTY_BKG,
                                      config.CALIBRATED_NASTY_BKG+config.CALIBRATED_NASTY_BKG_ERROR,
                                      config.CALIBRATED_NASTY_BKG-config.CALIBRATED_NASTY_BKG_ERROR,
                                      ],
                          ):

    shift_phs = {
                 self.tau_energy_init: tau_energy[0],
                 self.tau_energy_shift: (tau_energy[1] - tau_energy[2]) / 2.,
                 self.jet_energy_init: jet_energy[0],
                 self.jet_energy_shift: (jet_energy[1] - jet_energy[2]) / 2.,
                 self.lep_energy_init: lep_energy[0],
                 self.lep_energy_shift: (lep_energy[1] - lep_energy[2]) / 2.,
                 self.soft_term_init: soft_term[0],
                 self.soft_term_shift: (soft_term[1] - soft_term[2]) / 2.,
                 self.nasty_bkg_init: nasty_bkg[0],
                 self.nasty_bkg_shift: (nasty_bkg[1] - nasty_bkg[2]) / 2.,
                 }

    b_nom = templates[('b', tau_energy[0], jet_energy[0], lep_energy[0], soft_term[0], nasty_bkg[0])]
    b_up = np.array([
              templates[('b', tau_energy[1], jet_energy[0], lep_energy[0], soft_term[0], nasty_bkg[0])],
              templates[('b', tau_energy[0], jet_energy[1], lep_energy[0], soft_term[0], nasty_bkg[0])],
              templates[('b', tau_energy[0], jet_energy[0], lep_energy[1], soft_term[0], nasty_bkg[0])],
              templates[('b', tau_energy[0], jet_energy[0], lep_energy[0], soft_term[1], nasty_bkg[0])],
              templates[('b', tau_energy[0], jet_energy[0], lep_energy[0], soft_term[0], nasty_bkg[1])],
                    ])
    b_dw = np.array([
              templates[('b', tau_energy[2], jet_energy[0], lep_energy[0], soft_term[0], nasty_bkg[0])],
              templates[('b', tau_energy[0], jet_energy[2], lep_energy[0], soft_term[0], nasty_bkg[0])],
              templates[('b', tau_energy[0], jet_energy[0], lep_energy[2], soft_term[0], nasty_bkg[0])],
              templates[('b', tau_energy[0], jet_energy[0], lep_energy[0], soft_term[2], nasty_bkg[0])],
              templates[('b', tau_energy[0], jet_energy[0], lep_energy[0], soft_term[0], nasty_bkg[2])],
                    ])

    s_nom = templates[('s', tau_energy[0], jet_energy[0], lep_energy[0], soft_term[0], nasty_bkg[0])]
    s_up = np.array([
              templates[('s', tau_energy[1], jet_energy[0], lep_energy[0], soft_term[0], nasty_bkg[0])],
              templates[('s', tau_energy[0], jet_energy[1], lep_energy[0], soft_term[0], nasty_bkg[0])],
              templates[('s', tau_energy[0], jet_energy[0], lep_energy[1], soft_term[0], nasty_bkg[0])],
              templates[('s', tau_energy[0], jet_energy[0], lep_energy[0], soft_term[1], nasty_bkg[0])],
              templates[('s', tau_energy[0], jet_energy[0], lep_energy[0], soft_term[0], nasty_bkg[1])],
                    ])
    s_dw = np.array([
              templates[('s', tau_energy[2], jet_energy[0], lep_energy[0], soft_term[0], nasty_bkg[0])],
              templates[('s', tau_energy[0], jet_energy[2], lep_energy[0], soft_term[0], nasty_bkg[0])],
              templates[('s', tau_energy[0], jet_energy[0], lep_energy[2], soft_term[0], nasty_bkg[0])],
              templates[('s', tau_energy[0], jet_energy[0], lep_energy[0], soft_term[2], nasty_bkg[0])],
              templates[('s', tau_energy[0], jet_energy[0], lep_energy[0], soft_term[0], nasty_bkg[2])],
                    ])
    # OLD VERSION
    # s_nom = templates[('s', tau_energy[0])]
    # s_up = np.array([templates[('s', tau_energy[1])]])
    # s_dw = np.array([templates[('s', tau_energy[2])]])

    # remove zeroes
    zero_filter = np.all([(s_nom != 0.), (b_nom != 0.)], axis=0)
    templates = {k: v[zero_filter] for k, v in templates.items()
                 if not ('pars' in k[0])}

    self.shape_phs = {self.b_nom: b_nom[zero_filter],
                      self.b_up: b_up[:, zero_filter],
                      self.b_dw: b_dw[:, zero_filter],
                      self.s_nom: s_nom[zero_filter],
                      self.s_up: s_up[:, zero_filter],
                      self.s_dw: s_dw[:, zero_filter],
                      **shift_phs}

    return templates

  def templates_from_json(self, json_path,
                          tau_energy=[1.00, 1.03, 0.97]):

    with open(json_path) as f:
      templates = json.load(f)

    templates = {literal_eval(k): v for k, v in templates.items()}
    templates = self.templates_from_dict(templates, tau_energy=tau_energy)

    return templates

  def asimov_data(self, par_phs={}, sess=None):

    if sess is None:
      sess = tf.get_default_session()
    asimov_data = sess.run(self.t_exp, {**par_phs, **self.shape_phs})
    if not self.multiple_pars:
      asimov_data = asimov_data[0]
    return asimov_data

  def asimov_hess(self, par_phs={}, sess=None):

    if sess is None:
      sess = tf.get_default_session()
    obs_phs = {self.obs: self.asimov_data(par_phs, sess=sess)}
    print(obs_phs)
    h_hess = sess.run(self.h_hess, {**par_phs, **obs_phs, **self.shape_phs})
    return FisherMatrix(h_hess, par_names=list(self.all_pars.keys()))

  def hessian_and_gradient(self, pars, par_phs={}, obs_phs={}, sess=None):

    if sess is None:
      sess = tf.get_default_session()

    pars = tuple(pars)
    nll, hess, grad = sess.run([self.h_nll, self.h_hess, self.h_grad],
                               feed_dict={**par_phs, **obs_phs,
                                          **self.shape_phs})

    indices = [list(self.all_pars.keys()).index(par) for par in pars]
    idx_subset = np.reshape(list(it.product(indices, indices)),
                            (len(pars), len(pars), -1)).T

    sub_hess = hess[:, idx_subset[0], idx_subset[1]]
    sub_grad = grad[:, indices]
    return nll, sub_hess, sub_grad
