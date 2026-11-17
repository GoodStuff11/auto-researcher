import sys,os,numpy as np
sys.path.insert(0,os.getcwd())
import fix_test as F
M3,R=F.M3,F.R
sec=M3.make_sector(4,3,10,8.0,-0.25,"obc","obc")
sg=M3.SG(sec)
for kind in ("nnbf_SG","che"):
    arm=F.build_arm_fix(sec,kind,16,0)
    st=M3.State(sec,arm,F.SPROJ[kind],sg,16384)
    phi,_=arm.forward(arm.theta0,16384)
    print(kind, F.energy_of(sec, st.psi(phi)))
    if kind=="che":
        th=np.load("../P4.1_cheap_su2_r5/results/L4x3xobcobc_N10_U8_tp-0.25_w16_che_drift4000_seed0_theta.npy")
        phi,_=arm.forward(th,16384); print("drift", F.energy_of(sec, st.psi(phi)))
