import numpy as np
import cmath

def uv2intdir(U,V,decl_mag,ang_rot):

	""""""
	"""Codigo original em MatLab"""
	"""Author: Roberto Fioravanti Carelli Fontes"""
	"""Depto. de Oceanografia Fisica IOUSP"""
	"""Laboratorio de Hidrodinamica Costeira (LHICO)"""
	""""""
	"""Versao em Python, abr/2016"""
	"""Carine de Godoi Rezende Costa"""
	"""Davidson Laboratory - Stevens Institute of Technology"""
	"""LHiCo - IOUSP"""
	""""""
	"""INVERSO DA FUNCAO INTDIR2UV.M Decompoem o vetor corrente definido pela intensidade e direcao (ref. Norte -> Este) considerando a declinacao magnetica e a rotacao do sistema de coordenadas"""
	""""""
	"""Uso: entre com intensidade, direcao,declinacao magnetica e orientacao do eixo Y, a partir do Norte (0 deg.). Por exemplo, Canal de Sao Sebastiao = 51 deg. A declinacao para oeste e' negativa, p.ex.: -18 deg."""
	""""""
	""" Se valor de ang_rot nao for suprido, e' assumido que nao ha' rotacao alem disso, se decl_mag tambem nao for suprido nao e' feito correcao magnetica"""

	vetor = []
	for i,j in zip(U,V):
		vetor.append(complex(i,j))

	vetor = np.asarray(vetor)

	# print("parte real = " + str(vetor.real))
	# print("parte imaginaria = " + str(vetor.imag))

	INT = np.abs(vetor)
	# print("INT = " + str(INT))

	DIR = []
	for d in vetor:
		DIR.append(cmath.phase(d))

	DIR = np.asarray(DIR)
	
	# print("DIR = " + str(DIR))

	DIR = DIR * 180 / np.pi
	# print("DIR = " + str(DIR))

	DIR = DIR - decl_mag + ang_rot
	# print("DIR = " + str(DIR))

	DIR = np.mod(90 - DIR, 360)
	# print("DIR = " + str(DIR))

	return INT, DIR

def intdir2uv(int, dir, decl_mag, ang_rot):

	""""""
	"""Codigo original em MatLab"""
	"""Author: Roberto Fioravanti Carelli Fontes"""
	"""Depto. de Oceanografia Fisica IOUSP"""
	"""Laboratorio de Hidrodinamica Costeira (LHICO)"""
	""""""
	"""Versao em Python, abr/2016"""
	"""Carine de Godoi Rezende Costa"""
	"""Davidson Laboratory - Stevens Institute of Technology"""
	"""LHiCo - IOUSP"""
	""""""
	"""FUNCAO INTDIR2UV.M Decompoem o vetor corrente definido pela intensidade e direcao (ref. Norte -> Este) considerando a declinacao magnetica e a rotacao do sistema de coordenadas"""
	""""""
	"""Uso: entre com intensidade, direcao,declinacao magnetica e orientacao do eixo Y, a partir do Norte (0 deg.). Por exemplo, Canal de Sao Sebastiao = 51 deg. A declinacao para oeste e' negativa, p.ex.: -18 deg."""
	""""""
	"""Se valor de ang_rot nao for suprido, e' assumido que nao ha' rotacao. Alem disso, se decl_mag tambem nao for suprido nao e' feito correcao magnetica"""


	# decompor o vetor
	dir = dir + decl_mag
	# print("dir = " + str(dir))
	dir = np.mod(dir, 360)
	# print("dir = " + str(dir))
	dir = dir * np.pi / 180
	# print("dir = " + str(dir))

	# inlinacao da linha de costa
	# alinhar o sistema com a costa
	ang_rot = ang_rot * np.pi / 180
	# print("ang_rot = " + str(ang_rot))

	u = int * np.sin(dir)	# aqui eu passo de NE para
	v = int * np.cos(dir)	# XY
	# print("u = " + str(u))
	# print("v = " + str(v))

	U = u * np.cos(ang_rot) - v * np.sin(ang_rot)	# aqui eu faco a rotacao
	V = u * np.sin(ang_rot) + v * np.cos(ang_rot)	# segundo o alinhamento da costa
	# print("U = " + str(U))
	# print("V = " + str(V))

	# from putcdf.f
	# VELU = U*COS(ANG) - V*SIN(ANG)
	# VELV = U*SIN(ANG) + V*COS(ANG)

	return U, V