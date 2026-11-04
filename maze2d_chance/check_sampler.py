from baseline import *
from probe import load
# Verify zero-strength custom sampler is bitwise legacy sampling for batch1.
torch.set_num_threads(2);e=gym.make('maze2d-large-v1').unwrapped;e.reset();bounds=np.load(ROOT/'normalization.npz');ch=Chance(e,bounds);d,epoch=load_model()
start=np.array([[1.,1.,0.,0.]]);goal=np.array([[3.,1.]])
norm=lambda x:(x-ch.omiddle.numpy())/ch.oscale.numpy();cond={0:torch.tensor(norm(start),device='cuda').float(),383:torch.tensor(norm(np.c_[goal,np.zeros((1,2))]),device='cuda').float()}
torch.manual_seed(11)
with torch.no_grad():s=d(cond,verbose=False).cpu().numpy()
z,_,_=sample(d,ch,start,goal,[11],[-1.])
print('sample_max_diff',np.max(abs(s-z)))
print('global',execute(e,ch.physical_actions(torch.tensor(s[0])).numpy(),start[0],goal[0],0.,1))
print('custom',execute(e,ch.physical_actions(torch.tensor(z[0])).numpy(),start[0],goal[0],0.,1))
