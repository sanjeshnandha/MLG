import numpy as np
np.set_printoptions(precision=4, suppress=True)
E=[(0,1),(1,2),(1,3),(2,3),(3,4)]
n=5
A=np.zeros((n,n))
for i,j in E: A[i,j]=A[j,i]=1
d=A.sum(1); At=A+np.eye(n); dt=At.sum(1)
Dm=np.diag(dt**-0.5); Ah=Dm@At@Dm
X=np.array([[1,0],[1,1],[1,0],[0,1],[0,1]],float)
print("d",d,"dt",dt); print("Ahat\n",Ah)
AX=Ah@X; print("AX\n",AX)
import itertools, sys
W0=np.array([[1.,-1.],[-1.,1.]])
W1=np.array([[-1.,1.],[1.,-1.]])
S=AX@W0; H=np.maximum(S,0); print("S=AXW0\n",S); print("H1\n",H)
M=Ah@H; print("M=AH\n",M)
Z=M@W1; print("Z\n",Z)
P=np.exp(Z)/np.exp(Z).sum(1,keepdims=True); print("P\n",P)
Y={0:1,4:0}; w={0:0.3,1:0.7}
L=sum(w[c]*-np.log(P[i,c]) for i,c in Y.items())/sum(w[c] for c in Y.values())
print("losses", {i:-np.log(P[i,c]) for i,c in Y.items()}, "L", L)
dZ=np.zeros_like(Z); W=sum(w[c] for c in Y.values())
for i,c in Y.items():
    y=np.eye(2)[c]; dZ[i]=w[c]*(P[i]-y)/W
print("dZ\n",dZ)
dW1=M.T@dZ; print("dW1\n",dW1)
dM=dZ@W1.T; print("dM\n",dM)
dH=Ah.T@dM; print("dH\n",dH)
dS=dH*(S>0); print("dS\n",dS)
dW0=AX.T@dS; print("dW0\n",dW0)
eta=0.5
W0n=W0-eta*dW0; W1n=W1-eta*dW1
print("W0n\n",W0n,"\nW1n\n",W1n)
S2=AX@W0n; H2=np.maximum(S2,0); Z2=Ah@H2@W1n; P2=np.exp(Z2)/np.exp(Z2).sum(1,keepdims=True)
L2=sum(w[c]*-np.log(P2[i,c]) for i,c in Y.items())/W
print("P2\n",P2,"L2",L2)
# oversmoothing
def cos(H):
    Zn=H/np.linalg.norm(H,axis=1,keepdims=True); S=Zn@Zn.T; return (S.sum()-n)/(n*(n-1))
for k in [1,2,4,8,16,32]:
    Hk=np.linalg.matrix_power(Ah,k)@X; print(k, cos(Hk).round(4), Hk.round(3).tolist())
ev=np.linalg.eigvalsh(Ah); print("eig",ev)
pi=np.sqrt(dt)/np.linalg.norm(np.sqrt(dt)); print("pi",pi, Ah@pi)
