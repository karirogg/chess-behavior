import numpy as np
import chess.pgn
from tqdm import tqdm

import pandas as pd

def get_first_k_moves(use_username = True, username= None, path = None, k = 5, max_games = 100000):
    if username == None and path == None:
        return "Error please specify either username or path to pgn"
    if use_username:
        filename = f'games_{username}.pgn'
        path = f'games/{filename}'
    pgn = open(path)
    df = pd.DataFrame(columns = ['planes','White','WhiteElo','Black','BlackElo','date'])
    n = 1
    for n in tqdm(range(max_games)):
        game = chess.pgn.read_game(pgn)
        #print(game)
        if game == None:
            break
        n+=1
        white_name = game.headers["White"]
        black_name = game.headers["Black"]
        white_elo = game.headers["WhiteElo"]
        black_elo = game.headers["BlackElo"]
        date = game.headers["Date"]
        if white_name !='' and black_name !='' and white_elo not in ['','?'] and black_elo not in ['','?']:
            planes = []
            moves = game.mainline_moves()
            for i,move in enumerate(moves):
                if i >= 2 * k:
                    break
                planes.append(str(move))

            if len(planes) < 2 * k:
                continue

            row = pd.DataFrame({'planes':[planes],'White':white_name,'Black':black_name,'WhiteElo':int(white_elo),'BlackElo':(black_elo),'date':date})
            df = pd.concat([df,row])
    return df

import os

def get_concat_k_moves(dir, k = 5, max_games = 100000):
    list_of_df = []

    for game_file in os.listdir(dir):
        path = f'./{dir}/{game_file}'
        new_df = get_first_k_moves(use_username = False, path = path, k = k, max_games = max_games)
        list_of_df.append(new_df)
    return pd.concat(list_of_df)

def df_to_numpy(df):
    planes = df.planes.to_numpy()
    white_elo = df.WhiteElo.to_numpy().astype(np.int64)
    black_elo = df.BlackElo.to_numpy().astype(np.int64)

    players_to_int_dict = dict()
    list_of_players = list(set(df.White).union(set(df.Black)))

    for i,player in enumerate(list_of_players):
        players_to_int_dict[player] = i
        
    white_targets = np.array([players_to_int_dict[player] for player in df.White.to_numpy()])
    black_targets = np.array([players_to_int_dict[player] for player in df.Black.to_numpy()])

    X_list = []
    y_list = []

    for i in tqdm(range(len(planes))):
        plane_array = np.array(planes[i])

        # y_list consists of [white_player_id, white_elo_rating, black_player_id, black_elo_rating]
        X_list.append(plane_array)
        y_list.append([white_targets[i],white_elo[i], black_targets[i],black_elo[i]])

    X = np.array(X_list)
    y = np.array(y_list)
    return X,y

import pickle

dir = input("Enter directory name: ")
max_games = int(input("Enter max number of games: "))

df = get_concat_k_moves(dir, 20, max_games)

X,y = df_to_numpy(df)

filename = f'X_{dir}_{max_games}.dat'
outfile = open(filename,'wb')
pickle.dump(X,outfile)
outfile.close()

filename = f'y_{dir}_{max_games}.dat'
outfile = open(filename,'wb')
pickle.dump(y,outfile)
outfile.close()