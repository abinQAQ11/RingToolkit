import RingToolkit as rt
# ----------------------------------------------------------------------------------------------------------------------
DL = rt.Drift("DL", 1.75)
D1 = rt.Drift('D1', 0.01)
D2 = rt.Drift("D2", 0.09)
D3 = rt.Drift("D3", 0.10)
D4 = rt.Drift("D4", 0.20)
D5 = rt.Drift("D5", 0.14)
D6 = rt.Drift("D6", 0.02)

SQFo = rt.Quadrupole("SQFo", 0.2, 5.737)
SQFi = rt.Quadrupole("SQFi", 0.2, 4.998)

SCo = rt.Sextupole("SCo", 0.02, -29.997)
SDo = rt.Sextupole("SDo", 0.10, -84.763)
SDi = rt.Sextupole("SDi", 0.10, -68.176)
SCi = rt.Sextupole("SCi", 0.02, -20.157)

DIP = rt.Dipole("DIP", 1.0, -1.348, bending_angle=15)

DBA = rt.Lattice(half_cell=[DL, SQFo, D1, SCo, D2, SDo, D3, DIP, D4, SDi, D5, SCi, D6, SQFi],
                 name="DBA", periodicity=12, energy=1.5e9)

# 计算辐射积分（目前色品算得还不是很准，在调试）
DBA.radiation_parameters()
# 计算共振驱动项（目前只计算三阶的，四阶的正在编写）
DBA.driving_parameters()
# 画lattice图
rt.plot_lattice(DBA)
# 画共振驱动项
rt.plot_rdts(DBA)
# 输出为opa格式的文件
rt.export_to_opa(DBA)
# 输出为elegant格式的文件（目前功能还不完善）
rt.export_to_elegant(DBA)
