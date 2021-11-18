import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from agent.agent import Agent
from environment.Building import Building
from utils import is_finish, state_preprocessing
import argparse
import numpy as np
import os
os.makedirs('./model_weights', exist_ok=True)
import time

action_dim = 4 # UP,DOWN,LOAD,UNLOAD

if torch.cuda.is_available():
    device = torch.device("cuda") 
else:
    device = 'cpu'

def main():
    
    epoch_s = time.time()
    global args
    parser = argparse.ArgumentParser('parameters')
    parser.add_argument('--test', type=bool, default=False, help="True if test, False if train (default: False)")
    parser.add_argument('--epochs', type=int, default=10001, help='number of epochs, (default: 1001)')
    parser.add_argument('--lr_rate', type=float, default=0.0003, help='learning rate (default : 0.0003)')
    parser.add_argument('--lift_num', type=int, default=2, help='number of elevators')
    parser.add_argument('--T_horizon', type=int, default=2048, help='number of steps at once')
    parser.add_argument('--K_epoch', type=int, default=10, help='number of train at once')
    parser.add_argument('--minibatch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--gamma', type=float, default=0.99, help='training gamma')
    parser.add_argument('--lmbda', type=float, default=0.95, help='training lmbda')
    parser.add_argument('--eps_clip', type=float, default=0.2, help='training eps_clip')
    parser.add_argument('--critic_coef', type=float, default=0.5, help='training ciritic_coef')
    parser.add_argument('--building_height', type=int, default=8, help='building height ')
    parser.add_argument('--max_people_in_floor', type=int, default=8, help='maximum people in one floor')
    parser.add_argument('--max_people_in_elevator', type=int, default=8, help='maximum people in one elevator')
    parser.add_argument('--add_people_prob', type=float, default=0.8, help='add people probability')
    parser.add_argument("--load_file", type=str, default = '', help = 'load initial parameters')
    parser.add_argument("--save_interval", type=int, default = 50, help = 'save interval')
    parser.add_argument("--print_interval", type=int, default = 50, help = 'print interval')
    parser.add_argument("--test_case", type=bool, default = False, help = '')
    parser.add_argument("--dir", type=str, default = False, help = '')
    args = parser.parse_args()
    with open(f"./model_weights/{args.dir}/log.txt", 'a') as f:
        f.write(f'args.test : {args.test}')
        f.write(f'args.epochs : {args.epochs}')
        f.write(f'args.lr_rate : {args.lr_rate}')
        f.write(f'args.lift_num : {args.lift_num}')
        f.write(f'args.building_height :{args.building_height}')
        f.write(f'args.max_people_in_floor : {args.max_people_in_floor }')
        f.write(f'args.max_people_in_elevator :{args.max_people_in_elevator}')
        f.write(f'args.load_file : {args.load_file}')
        f.write(f'args.save_interval :{args.save_interval}')
        f.write(f'args.print_interval :{args.print_interval}')
        f.write(f'args.test_case : {args.test_case}')
        
    print('args.test : ', args.test)
    print('args.epochs : ', args.epochs)
    print('args.lr_rate : ', args.lr_rate)
    print('args.lift_num : ', args.lift_num)
    print('args.building_height :', args.building_height)
    print('args.max_people_in_floor : ', args.max_people_in_floor )
    print('args.max_people_in_elevator :', args.max_people_in_elevator)
    print('args.load_file : ', args.load_file)
    print('args.save_interval :', args.save_interval)
    print('args.print_interval :', args.print_interval)
    print('args.test_case : ', args.test_case)

    building = Building(args.lift_num, args.building_height, args.max_people_in_floor,\
                        args.max_people_in_elevator)
    
    agent = Agent(device,args.lift_num, args.building_height, args.max_people_in_floor,\
                        args.max_people_in_elevator,action_dim,args.K_epoch,\
                        args.gamma,args.lmbda,args.lr_rate,args.eps_clip,args.critic_coef,args.minibatch_size)
    if args.load_file:
        agent.load_state_dict(torch.load(args.load_file, map_location=device))
    summary = SummaryWriter()
    if torch.cuda.is_available():
        agent.cuda()
        print('cuda on')
    building.empty_building()
    while building.remain_passengers_num == 0 :
        # building.get_passengers(args.add_people_prob)
        building.generate_passengers(args.add_people_prob)
    floor_state,elv_state,elv_place_state = building.get_state()
    floor_state,elv_state,elv_place_state = state_preprocessing(args,device,floor_state,elv_state,elv_place_state)
    done = False
    global_step = 0
    score = 0.0
    score_lst = []
    for epoch in range(args.epochs):
        for t in range(args.T_horizon):
            global_step += 1
            action_prob = agent.get_action(floor_state,elv_state,elv_place_state)[0]
            m = Categorical(action_prob)
            action = m.sample().tolist()
            reward = building.perform_action(action)
            next_floor_state,next_elv_state,next_elv_place_state = building.get_state()

            next_floor_state,next_elv_state,next_elv_place_state = state_preprocessing(args,device,next_floor_state,next_elv_state,next_elv_place_state)
            done = is_finish((next_floor_state,next_elv_state))
            if done == True:
                reward = 100.
            else:
                reward = -10.
            agent.put_data((floor_state.cpu().tolist(),\
                            elv_state.cpu().tolist(),\
                            elv_place_state.cpu().tolist(),\
                            action, reward/100.0, \
                            next_floor_state.cpu().tolist(),\
                            next_elv_state.cpu().tolist(), \
                            next_elv_place_state.cpu().tolist(),\
                            [action_prob[idx][action[idx]] for idx in range(len(action_prob))],\
                            done))
            score += reward
            if args.test:
                os.system("cls")
                building.print_building(global_step)
                print(action)
                print('now reward : ', building.cumulated_reward)

                time.sleep(1.5)
            if (global_step > 300):
                done = True
            if done : 
                score_lst.append(score)
                summary.add_scalar('reward', score, epoch)
                score = 0
                global_step = 0
                building.empty_building()
                while building.remain_passengers_num == 0 :
                    # building.get_passengers(args.add_people_prob)     # modified
                    building.generate_passengers(args.add_people_prob) # original
                floor_state,elv_state,elv_place_state = building.get_state()
                floor_state,elv_state,elv_place_state = state_preprocessing(args,device,floor_state,elv_state,elv_place_state)
            else:
                floor_state = next_floor_state
                elv_state = next_elv_state
                elv_place_state = next_elv_place_state

        agent.train(summary,epoch)

        if epoch%args.print_interval==0 and epoch!=0:
            print("# of episode :{}, avg score : {:.1f}".format(epoch, sum(score_lst)/len(score_lst)), end=' ')
            epoch_e = time.time()
            secs = epoch_e-epoch_s
            print(f"epoch 소요시간:\t{int(secs//60)}분 {int(secs - 60)}초")
            score_lst = []
            epoch_s = time.time()
            with open(f"./model_weights/{args.dir}/log.txt", 'a') as f:
                try:
                    f.write("# of episode :{}, avg score : {:.1f}".format(epoch, sum(score_lst)/len(score_lst)))
                except:
                    pass

        if (epoch % args.save_interval == 0 )& (epoch != 0):
            if args.dir:
                os.makedirs(f'./model_weights/{args.dir}', exist_ok=True)
                torch.save(agent.state_dict(), f'./model_weights/{args.dir}/model_'+str(epoch))
            else:
                torch.save(agent.state_dict(), './model_weights/model_'+str(epoch))



if __name__ == '__main__':
    s = time.time()
    main()
    e = time.time()
    secs = e-s
    print(f"총 소요시간:\t{secs//60}분 {secs%60}초")