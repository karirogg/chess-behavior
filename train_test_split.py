import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm

def get_n_most_frequent_players(y, n = 100):
    all_occurances = np.concatenate([y[:,0], y[:,2]])
    counts = np.bincount(all_occurances)
    most_freq_players = np.argpartition(counts, -n)[-n:]
    return most_freq_players

def get_players_with_n_games(y, n = 100):
    all_occurances = np.concatenate([y[:,0], y[:,2]])
    counts = np.bincount(all_occurances)
    legal_players = np.where(counts >= n)[0]

    return legal_players

def get_embed_data(X, y, players, k):
    X_for_embeds_list = []
    y_for_embeds_list = []

    X_test_list = []
    y_test_list = []

    test_inds = []

    for player in tqdm(players):
        player_white_games_inds = y[:,0] == player
        player_black_games_inds = y[:,2] == player

        seen_test_inds = np.random.choice(np.arange(2*len(y))[np.concatenate([player_white_games_inds, player_black_games_inds])], 2*k, replace = False)

        inds_for_embeds = seen_test_inds[:k]
        inds_for_test = seen_test_inds[k:]

        seen_embed_inds_white = inds_for_embeds[inds_for_embeds < len(player_white_games_inds)]
        seen_embed_inds_black = inds_for_embeds[inds_for_embeds >= len(player_white_games_inds)] - len(player_white_games_inds)

        seen_test_inds_white = inds_for_test[inds_for_test < len(player_white_games_inds)]
        seen_test_inds_black = inds_for_test[inds_for_test >= len(player_white_games_inds)] - len(player_white_games_inds)

        X_for_embeds_list.append(np.concatenate((X[seen_embed_inds_white], X[seen_embed_inds_black]), axis = 0))
        y_for_embeds_list.append(np.concatenate((y[seen_embed_inds_white, 0], y[seen_embed_inds_black, 2]), axis = 0))

        X_test_list.append(np.concatenate((X[seen_test_inds_white], X[seen_test_inds_black]), axis = 0))
        y_test_list.append(np.concatenate((y[seen_test_inds_white, 0], y[seen_test_inds_black, 2]), axis = 0))
        
        test_inds.append(np.concatenate((seen_embed_inds_white, seen_embed_inds_black, seen_test_inds_white, seen_test_inds_black)))

        # X_for_embeds_list.append(X[seen_test_inds])

        # X_for_embeds_list.append(X[seen_test_inds[:k]])
        # y_for_embeds_list.append(y[seen_test_inds[:k]])

        # X_test_list.append(X[seen_test_inds[k:]])
        # y_test_list.append(y[seen_test_inds[k:]])

    # delete games used for test/embeddings from the seen train set
    X = np.delete(X, np.unique(np.concatenate(test_inds)), axis = 0)
    y = np.delete(y, np.unique(np.concatenate(test_inds)), axis = 0)

    X_for_embeds = np.concatenate(X_for_embeds_list, axis = 0)
    y_for_embeds = np.concatenate(y_for_embeds_list, axis = 0)

    X_test = np.concatenate(X_test_list, axis = 0)
    y_test = np.concatenate(y_test_list, axis = 0)

    return X, y, X_for_embeds, y_for_embeds, X_test, y_test

def get_all_splits(X, y, k = 50):
    seen_players = get_n_most_frequent_players(y, 400)

    most_played_inds = np.logical_or(np.isin(y[:,0], seen_players), np.isin(y[:,2], seen_players))
    X_seen = X[most_played_inds]
    y_seen = y[most_played_inds] # leikir þeirra sem spila mest

    X_seen, y_seen, X_seen_for_embeds, y_seen_for_embeds, X_seen_test, y_seen_test = get_embed_data(X_seen, y_seen, seen_players, k)

    X_seen_train, X_seen_val, y_seen_train, y_seen_val = train_test_split(X_seen, y_seen, test_size = 0.2, random_state = 42)

    X_rest = X[np.logical_not(most_played_inds)] # hinir leikirnir
    y_rest = y[np.logical_not(most_played_inds)]

    # Finnur leikmenn sem hafa n leiki
    unseen_players = get_players_with_n_games(y_rest, n = 2*k) # Take these from the train and validation sets

    unseen_inds = np.logical_or(np.isin(y_rest[:,0], unseen_players), np.isin(y_rest[:,2], unseen_players))
    X_unseen = X_rest[unseen_inds]
    y_unseen = y_rest[unseen_inds]

    _, _, X_unseen_for_embeds, y_unseen_for_embeds, X_unseen_test, y_unseen_test = get_embed_data(X_unseen, y_unseen, unseen_players, k)  

    return seen_players, unseen_players, X_seen_train, y_seen_train, X_seen_val, y_seen_val, X_seen_for_embeds, y_seen_for_embeds, X_seen_test, y_seen_test, X_unseen_for_embeds, y_unseen_for_embeds, X_unseen_test, y_unseen_test

dir = input("Enter directory: ")
max_games = int(input("Enter max games: "))
k = int(input("Enter k (k-shot learning): "))

X = pickle.load(open(f'X_{dir}_{max_games}.dat','rb'))
y = pickle.load(open(f'y_{dir}_{max_games}.dat','rb'))

print(X.shape)

# seen_players, unseen_players, X_seen_train, y_seen_train, X_seen_val, y_seen_val, X_seen_for_embeds, y_seen_for_embeds, X_seen_test, y_seen_test, X_unseen_for_embeds, y_unseen_for_embeds, X_unseen_test, y_unseen_test
player_data = get_all_splits(X,y, k = k)

filename = f'training_data_{dir}_{max_games}_{k}2.dat'
outfile = open(filename,'wb')
pickle.dump(player_data,outfile)
outfile.close()