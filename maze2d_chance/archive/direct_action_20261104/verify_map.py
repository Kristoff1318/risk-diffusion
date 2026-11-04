"""Read-only map verification and common-coordinate rendering; no executions."""
from baseline import *
from diffuser.datasets import load_environment
from diffuser.utils.rendering import Maze2dRenderer
from d4rl.pointmaze.maze_model import LARGE_MAZE,LARGE_MAZE_EVAL,parse_maze
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

native=gym.make('maze2d-large-v1').unwrapped
legacy=load_environment('maze2d-large-v1')
renderer=Maze2dRenderer('maze2d-large-v1')
config=pickle.load(open(CHECKPOINT/'dataset_config.pkl','rb'))
renderconfig=pickle.load(open(CHECKPOINT/'render_config.pkl','rb'))
assert config._dict['env']=='maze2d-large-v1'
assert np.array_equal(native.maze_arr,legacy.maze_arr)
assert np.array_equal(native.maze_arr,renderer.env.maze_arr)
assert np.array_equal(native.maze_arr,parse_maze(LARGE_MAZE))
assert not np.array_equal(native.maze_arr,parse_maze(LARGE_MAZE_EVAL))
# Verify EVERY native MuJoCo wall has its expected LARGE_MAZE cell index,
# including the body-local physical qpos offset used in the original report.
indices=np.argwhere(native.maze_arr==10)
geoms=np.array([native.model.geom_pos[i,:2] for i in range(native.model.ngeom) if (native.model.geom_id2name(i) or '').startswith('wall_')])
assert np.array_equal(geoms,indices+1.)
centers=geoms-native.model.body_pos[1,:2]
assert np.allclose(centers,indices-.2)
sha=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
registration=gym.spec('maze2d-large-v1')
evidence=dict(registration_id=registration.id,entry_point=registration.entry_point,registration_is_large_maze=registration._kwargs['maze_spec']==LARGE_MAZE,checkpoint_dataset=config._dict['env'],checkpoint_renderer_config=renderconfig._dict,array_shape=list(native.maze_arr.shape),wall_count=len(indices),array_sha256=sha(native.maze_arr),legacy_loader_exact_match=True,legacy_renderer_exact_match=True,is_LARGE_MAZE=True,is_LARGE_MAZE_EVAL=False,all_native_wall_geometries_match=True,native_body_offset=native.model.body_pos[1,:2].tolist(),physical_qpos_wall_centers='maze_indices - 0.2',original_display='horizontal qpos_x; vertical qpos_y increasing upward; boundary crop',corrected_display='horizontal qpos_y; vertical qpos_x increasing downward; full boundaries; equal physical aspect',legacy_display='horizontal (qpos_y+0.5)/12; vertical (qpos_x+0.5)/9 increasing downward; square normalized aspect')
(ROOT/'results'/'map_verification.json').write_text(json.dumps(evidence,indent=2,default=str))
fig,axes=plt.subplots(1,3,figsize=(15,5))
# Legacy canonical cell display, then same cell mask read through the experiment,
# then exact physical geoms in the same orientation. No new simulation.
for ax,arr,title in [(axes[0],renderer.env.maze_arr,'Legacy RB Diffuser loader'),(axes[1],native.maze_arr,'Experiment native maze2d-large-v1')]:
 ax.imshow(arr==10,cmap='Greys',vmin=0,vmax=1,origin='upper',extent=(-.5,11.5,8.5,-.5),interpolation='nearest');ax.set_title(title);ax.set_xlabel('maze second index / horizontal');ax.set_ylabel('maze first index / downward');ax.set_aspect('equal')
ax=axes[2]
for p in centers:ax.add_patch(Rectangle((p[1]-.5,p[0]-.5),1,1,facecolor='.35',edgecolor='white',lw=.4))
ax.set_xlim(-.7,11.3);ax.set_ylim(8.3,-.7);ax.set_aspect('equal');ax.set_title('Verified physical geoms\ncorrected familiar orientation');ax.set_xlabel('qpos_y');ax.set_ylabel('qpos_x / downward')
fig.suptitle('Exact same LARGE_MAZE array: 9 × 12; not LARGE_MAZE_EVAL',fontsize=14);fig.tight_layout();fig.savefig(ROOT/'results'/'map_verification.png',dpi=150)
print(json.dumps(evidence,indent=2,default=str))
