"""Visual comparison by replaying already-audited noise draws, not new outcomes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
from environment import make_environment
from evaluate_tracking import execute_reference

ROOT=Path(__file__).parent
OUT=ROOT/'results/chance_comparison'


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    rows=json.loads((OUT/'results.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(15,9),sharex=True,sharey=True)
    evidence=[]
    critical_position=None
    # Seed1 is the wall-only risk-repair case and its same-start obstacle stress case.
    for i,(scenario,obstacles) in enumerate(protocol['scenarios'].items()):
        env=make_environment(obstacles)
        centers=[env.model.geom_pos[g,:2]-env.model.body_pos[1,:2] for g in range(env.model.ngeom)
                 if (env.model.geom_id2name(g) or '').startswith('wall_')]
        for j,budget in enumerate([None]+protocol['budgets']):
            row=next(r for r in rows if r['scenario']==scenario and r['seed']==1 and r['budget']==budget)
            saved=np.load(OUT/f'{row["name"]}_paths.npz');reference=saved['reference']
            owner=row['exact_path_audit_reuse'] or row['name']
            flags=np.load(OUT/f'{owner}_executions.npz')['audit_flags']
            ax=axes[i,j]
            for p in centers:ax.add_patch(Rectangle((p[1]-.5,p[0]-.5),1,1,facecolor='.72'))
            for x,y,radius in obstacles:ax.add_patch(Circle((y,x),radius,color='firebrick',alpha=.75))
            traces=[]
            for draw in range(32):
                trial=execute_reference(env,reference,np.array(row['start']),np.array(row['goal']),
                    protocol['noise_std'],protocol['audit_seed_base']+i*1000+100+draw,save=True)
                actual=np.array([trial['wall'],trial['added_obstacle'],trial['goal_failure']])
                assert np.array_equal(actual,flags[draw]),(row['name'],draw)
                path=trial['trajectory'];traces.append(path)
                color='crimson' if actual.any() else 'tab:blue'
                ax.plot(path[:,1],path[:,0],color=color,alpha=.55 if actual.any() else .15,
                        lw=1.1 if actual.any() else .8,zorder=3 if actual.any() else 2)
            np.savez_compressed(OUT/f'{row["name"]}_visual_replay.npz',trajectories=np.array(traces),audit_flags=flags[:32])
            if i==0:
                if j==0:
                    failed_draw=np.flatnonzero(flags[:32].any(1))[0]
                    candidate=traces[failed_draw][:,:2]
                    delta=candidate[:,None,:]-np.asarray(centers)[None,:,:]
                    box=np.abs(delta)-.5
                    margin=np.linalg.norm(np.maximum(box,0),axis=-1)+np.minimum(box.max(-1),0)-.102
                    critical_position=candidate[np.argmin(margin.min(-1))]
                inset=ax.inset_axes([.57,.59,.39,.35])
                for center in centers:
                    inset.add_patch(Rectangle((center[1]-.5,center[0]-.5),1,1,facecolor='.72'))
                for draw,path in enumerate(traces):
                    failed=flags[draw].any()
                    inset.plot(path[:,1],path[:,0],color='crimson' if failed else 'tab:blue',
                               alpha=.55 if failed else .15,lw=1.,zorder=3 if failed else 2)
                inset.plot(reference[:,1],reference[:,0],'--',color='black',lw=.8,zorder=4)
                inset.set_xlim(critical_position[1]-.18,critical_position[1]+.18)
                inset.set_ylim(critical_position[0]+.18,critical_position[0]-.18)
                inset.set_aspect('equal');inset.set_xticks([]);inset.set_yticks([])
                inset.set_title('same corner, enlarged',fontsize=7)
                rectangle,_=ax.indicate_inset_zoom(inset,edgecolor='black',alpha=.5)
                rectangle.set_label('_nolegend_')

            ax.plot(reference[:,1],reference[:,0],'--',color='black',lw=1.1,label='reference')
            goal=row['goal'];start=row['start']
            ax.add_patch(Circle((goal[1],goal[0]),.5,color='green',alpha=.17));ax.plot(start[1],start[0],'go',ms=4)
            label='Nominal' if budget is None else f'Chance guidance: {budget:.0%} budget'
            ax.set_title(f'{label}\nindependent audit: {row["audit"]["failures"]}/256 failures',fontsize=11)
            ax.set_xlim(-.7,11.3);ax.set_ylim(8.3,-.7);ax.set_aspect('equal')
            if j==0:ax.set_ylabel('Walls only' if i==0 else 'Walls + added obstacles')
            evidence.append(dict(name=row['name'],replayed_existing_draws=32,all_flags_exact=True))
        env.close()
    axes[0,0].plot([],[],color='tab:blue',label='safe noisy execution')
    axes[0,0].plot([],[],color='crimson',label='failed noisy execution')
    axes[0,0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Same start (seed1), original controller: nominal vs slide-based guidance\n32 existing audit draws per panel; all three starts remain in the full report',fontsize=13)
    fig.tight_layout();fig.savefig(OUT/'comparison_overview.png',dpi=160);plt.close(fig)
    (OUT/'visual_replay_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print('Rendered comparison_overview.png; all 192 replayed outcomes match saved audit flags.')


if __name__=='__main__':main()
