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
        new_df = get_first_k_moves(use_username = False, path = path, max_games = max_games)
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

def generate_X_y(dir, max_games):
    df = get_concat_k_moves(dir, 20, max_games)#, target_date = '2012')
    #print(df.shape)
    X,y = df_to_numpy(df)

    filename = 'X.dat'
    outfile = open(filename,'wb')
    pickle.dump(X,outfile)
    outfile.close()
    
    filename = 'y.dat'
    outfile = open(filename,'wb')
    pickle.dump(y,outfile)
    outfile.close()
    
    return X,y

def get_n_most_frequent_players(y, n = 100):
    all_occurances = np.concatenate([y[:,0], y[:,2]])
    counts = np.bincount(all_occurances)
    most_freq = np.argpartition(counts, -n)[-n:]
    return most_freq

def get_players_with_n_games(y, n = 100):
    all_occurances = np.concatenate([y[:,0], y[:,2]])
    counts = np.bincount(all_occurances)
    legal_players = np.where(counts >= n)
    return legal_players

from sklearn.model_selection import train_test_split
def get_all_splits(X, y):
    most_freq = get_n_most_frequent_players(y, 100)

    most_played_inds = np.logical_or(y[:,0].isin(most_freq), y[:,2].isin(most_freq))
    X_most_played = X[most_played_inds]
    y_most_played = y[most_played_inds] # leikir þeirra sem spila mest

    X_rest = X[np.logical_not(most_played_inds)] # hinir leikirnir
    y_rest = y[np.logical_not(most_played_inds)]

    # KR rage-ar út í þetta
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_most_played, y_most_played, test_size = 0.2, random_state = 2)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size = 0.125, random_state = 2) # 70 % train, 10 % val, 20 % test

    # Finnur leikmenn sem hafa n leiki
    inds_for_embeds = get_players_with_n_games(y_rest) # Take these from the train and validation sets

    X_train_for_embeds = X_rest[inds_for_embeds]
    y_train_for_embeds = y_rest[inds_for_embeds]

    return X_train, y_train, X_val, y_val, ...


dir = input("Enter directory to read data: ")
max_games = int(input("Enter the max number of games per file you want to read: "))
X, y = generate_X_y(dir = dir, max_games = max_games)
print(X, y)
print(X.shape, y.shape)

