"""Reference screenshot comparison; image analysis and saved paths only."""
from baseline import *
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diffuser.utils.rendering import Maze2dRenderer
REFERENCE=Path('/home/kristoff/.codex/attachments/4f5c88fb-c916-4e9b-955c-999491bdbd48/codex-clipboard-30f5365b-9417-48d9-b973-e18f778b9d55.png')
rgb=np.array(Image.open(REFERENCE).convert('RGB'))
# The visible field occupies x=[77,539), y=[72,535), with 12 columns,9 rows.
# Exclude the legend, then classify each cell by its remaining flat gray-wall area.
x0,y0,x1,y1=77,72,539,535
flatgray=(rgb.max(-1)-rgb.min(-1)<=2)&(rgb[:,:,0]>=120)&(rgb[:,:,0]<=135)
valid=np.ones(rgb.shape[:2],bool);valid[79:145,399:536]=False
observed=np.zeros((9,12),bool);fractions=np.zeros((9,12))
for i in range(9):
 for j in range(12):
  ya=round(y0+i*(y1-y0)/9)+2;yb=round(y0+(i+1)*(y1-y0)/9)-2
  xa=round(x0+j*(x1-x0)/12)+2;xb=round(x0+(j+1)*(x1-x0)/12)-2
  keep=valid[ya:yb,xa:xb];fractions[i,j]=flatgray[ya:yb,xa:xb][keep].mean();observed[i,j]=fractions[i,j]>.5
renderer=Maze2dRenderer('maze2d-large-v1');expected=renderer.env.maze_arr==10
mismatches=np.argwhere(observed!=expected)
print('MISMATCHES',mismatches,'wallcounts',observed.sum(),expected.sum(),flush=True)
assert len(mismatches)==0
out=ROOT/'results'
evidence=dict(reference_path=str(REFERENCE),reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),field_pixel_bounds=[x0,y0,x1,y1],excluded_legend_pixel_bounds=[399,79,536,145],threshold='flat gray RGB 120..135, channel range<=2; cell wall if >50% eligible interior pixels',detected_wall_count=int(observed.sum()),native_wall_count=int(expected.sum()),compared_cells=108,mismatch_cells=mismatches.tolist(),all_108_cells_match=True,cell_width_pixels=(x1-x0)/12,cell_height_pixels=(y1-y0)/9,reference_cell_aspect='cell height is approximately 4/3 cell width; normalized square field stretches 9 physical rows vs12 columns',added_obstacles='Three red added obstacles appear in reference only; absent from current experiment and not overlaid as evaluated obstacles.')
(out/'reference_comparison.json').write_text(json.dumps(evidence,indent=2))
# Call actual legacy renderer on a recorded native no-noise path. This creates
# the legacy square stretch and historical +.5 overlay convention exactly.
p=np.load(out/'nominal_-1.0_0_rollouts.npz')['nominal']
legacy_img=renderer.renders(p.copy(),title='Saved native rollout; no added obstacles')
Image.fromarray(legacy_img).save(out/'trajectories_legacy_square.png')
fig,ax=plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(rgb);ax[0].axis('off');ax[0].set_title('User reference\nthree added red obstacles')
ax[1].imshow(observed,cmap='gray_r',vmin=0,vmax=1,extent=(0,1,1,0),interpolation='nearest');ax[1].axis('off');ax[1].set_title('Extracted 9×12 native wall mask\n108/108 cells match current environment')
ax[2].imshow(legacy_img);ax[2].axis('off');ax[2].set_title('Actual legacy renderer + recorded path\nno added obstacles in current experiment')
fig.suptitle('Same native walls; reference uses square normalization and additional red obstacles',fontsize=13)
fig.tight_layout();fig.savefig(out/'reference_comparison.png',dpi=150)
print(json.dumps(evidence,indent=2))
