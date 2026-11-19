import argparse, json, time
import cv2, gym, gym_unrealcv

ap=argparse.ArgumentParser(); ap.add_argument('--task', required=True); ap.add_argument('--out', default='/workspace/runs/uavflow_diag'); a=ap.parse_args()
task=json.load(open(a.task)); p=[float(v) for v in task['initial_pos']]
env=gym.make('UnrealTrack-DowntownWest-ContinuousColor-v0'); u=env.unwrapped; u.agents_category=['drone']
if not u.launched:
    u.launched=u.launch_ue_env()
player=next((p for p in u.player_list if "Drone" in p), u.player_list[0])
cam=u.cam_list[u.player_list.index(player)]
u.unrealcv.set_viewport(player)
u.unrealcv.set_phy(player, 0)
def snap(tag):
    img=u.unrealcv.get_image(cam,'lit'); loc=u.unrealcv.get_obj_location(player); rot=u.unrealcv.get_obj_rotation(player)
    cv2.imwrite(f'{a.out}/{tag}.png', img)
    print(tag, 'loc', loc, 'rot', rot, 'shape', img.shape, flush=True)
    return img
u.unrealcv.set_obj_location(player,p[:3]); u.unrealcv.set_rotation(player,p[4]-180); u.unrealcv.set_cam(cam,[p[0],p[1],p[2]+60],[p[3],p[4],p[5]]); time.sleep(1)
i0=snap('a')
q=[p[0]+5,p[1],p[2]]
u.unrealcv.set_obj_location(player,q); u.unrealcv.set_rotation(player,p[4]-180); u.unrealcv.set_cam(cam,[q[0],q[1],q[2]+60],[p[3],p[4],p[5]]); time.sleep(1)
i1=snap('b')
import numpy as np
print('diff', float(np.mean(cv2.absdiff(i0,i1))), flush=True)
