from baseline import *
e=gym.make('maze2d-large-v1').unwrapped;e.reset();ch=Chance(e,np.load(ROOT/'normalization.npz'))
for start,goal in TASKS:
 p=np.load(ROOT/('probe_'+str(goal)+'.npz'));a=p['actions'];z=torch.zeros(1,384,6);z[0,:,:2]=torch.tensor((a-ch.amiddle.numpy())/ch.ascale.numpy())
 print(start,goal,ch.risk(z,torch.tensor([start]),torch.tensor([goal]))[:3])
