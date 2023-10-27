from os import listdir, environ
from os.path import isdir, join, splitext, dirname

# Plotting
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd
from statannotations.Annotator import Annotator

# Utility
import sys
import math
import time
from collections import defaultdict
import pprint

# Parse simulation log
sys.path.append(join(dirname(__file__), "..", "protos", "generated")) # Path to compiled proto files
import time_step_pb2
from load_data import SimData


# Path to simulation logs
RESULTS_DIR = join(environ['HOME'], 'multi-human-swarm-control/user_study_logs')
BINARY_FILENAME = 'log_data.pb'
COMMANDS_FILENAME = 'commands.csv'

# Path to questionnaire results
CSV_FILENAME = join(environ['HOME'], 'multi-human-swarm-control/user_study_questionnaires', 'questionnaire_results.csv')


def get_config(cond, send):
    cond_value = ''
    send_value = ''

    if cond[0:3] == 'dir':
        if send == 'send':
            cond_value = 'DIR'
            send_value = 'RS'
        elif send == 'no_send':
            cond_value = 'DIR'
            send_value = 'NRS'
        else:
            print('not send or no_send')
    elif cond[0:3] == 'ind':
        if send == 'send':
            cond_value = 'IND'
            send_value = 'RS'
        elif send == 'no_send':
            cond_value = 'IND'
            send_value = 'NRS'
        else:
            print('not send or no_send')
    else:
        print('not dir or ind')

    return cond_value, send_value


def load_logs():

    cond_dirs = [f for f in listdir(RESULTS_DIR) if isdir(join(RESULTS_DIR, f))]
    cond_dirs.sort()

    print('Parsing experiments ...')

    start_time = time.time()
    stats = {}

    # Initialize dict/list for each experiment condition
    for cond_dir in cond_dirs: # {dir1, dir2, ind1, ind2}

        stats[cond_dir] = {} 

        send_dirs = [f for f in listdir(join(RESULTS_DIR, cond_dir)) if isdir(join(RESULTS_DIR, cond_dir, f))]
        send_dirs.sort()

        for send_dir in send_dirs: # {'no_send', 'send'}
            stats[cond_dir][send_dir] = []

    # Load each trial data. Store them according to their experiment conditions
    for cond_dir in cond_dirs:

        send_dirs = [f for f in listdir(join(RESULTS_DIR, cond_dir)) if isdir(join(RESULTS_DIR, cond_dir, f))]
        send_dirs.sort()

        for send_dir in send_dirs:

            trial_dirs = [f for f in listdir(join(RESULTS_DIR, cond_dir, send_dir)) if isdir(join(RESULTS_DIR, cond_dir, send_dir, f))]
            trial_dirs.sort()

            for trial_dir in trial_dirs:

                start_time_single = time.time()

                # Load log and commands file
                log_file = join(RESULTS_DIR, cond_dir, send_dir, trial_dir, BINARY_FILENAME) # path: RESULTS_DIR/cond_dir/send_dir/trial_dir/BINARY_FILE
                commands_file = join(RESULTS_DIR, cond_dir, send_dir, trial_dir, COMMANDS_FILENAME) # path: RESULTS_DIR/cond_dir/send_dir/trial_dir/COMMANDS_FILE
                s = SimData(log_file, commands_file)
                stats[cond_dir][send_dir].append(s)

                duration_single = round(time.time() - start_time_single, 3)
                duration_total = round(time.time() - start_time, 3)
                print("Loaded -- '{0}' --\tin {1} s ({2} s)".format(join(send_dir, trial_dir), duration_single, duration_total))

    duration = round(time.time() - start_time, 3)
    print('Finished loading in {0} seconds'.format(duration))

    return stats


def load_questionnaires():
    
    df = pd.read_csv(CSV_FILENAME)

    # # Split data by experiment order
    # order1_df = df[df['Order'] == 1]
    # order2_df = df[df['Order'] == 2]

    # # Rename column labels by sending config (1: no send, 2: send)
    # column_labels_1 = list(order1_df)
    # column_labels_1 = [label + '-NS' if label[0] == 'Q' and label[-2] != '.' and int(label[1:]) < 18 else label for label in column_labels_1]
    # column_labels_1 = [label.replace('.1','-S') if label[0] == 'Q' else label for label in column_labels_1]
    # order1_df.columns = column_labels_1

    # column_labels_2 = list(order2_df)
    # column_labels_2 = [label + '-S' if label[0] == 'Q' and label[-2] != '.' and int(label[1:]) < 18 else label for label in column_labels_2]
    # column_labels_2 = [label.replace('.1','-NS') if label[0] == 'Q' else label for label in column_labels_2]
    # order2_df.columns = column_labels_2

    # # Join the data back together
    # frames = [order1_df, order2_df]
    # df = pd.concat(frames)

    return df


def plot_points_scored(stats, include_partial_points=False):
    """Plot points scored in each condition (dir_no_send, dir_send, ind_no_send, ind_send)"""

    df = pd.DataFrame({
                        'Communication': pd.Series(dtype='str'), 
                        'Condition':     pd.Series(dtype='str'), 
                        'Users':         pd.Series(dtype='str'),
                        'Points':        pd.Series(dtype='float')
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:
                print(trial.users, trial.totalPoints)

                #Store result to dataframe
                cond_value, send_value = get_config(cond, send)
                points_scored = trial.totalPoints

                if include_partial_points:

                    partial_points_scored = 0

                    # Check final timestep for any partially completed tasks
                    for task in trial.data[trial.totalTime]['log'].tasks:
                        factor = (300. - 50) / (12. - 1)
                        init_demand = math.floor((task.requiredRobots * factor) + (50 - factor))

                        if task.demand < init_demand:
                            # Calculate points for each unfinished task
                            partial_points_scored += (float(task.demand) / init_demand) * task.requiredRobots
                            # print('Partially completed task of size', task.requiredRobots, '(', task.demand, '/', init_demand, ') Adding', (float(task.demand) / init_demand) * task.requiredRobots)

                    print('Total partial points', partial_points_scored)

                    # Add partially completed tasks to the score
                    points_scored += partial_points_scored

                # Add average distance traveled to dataframe
                d = pd.DataFrame({'Communication': [cond_value], 'Condition': [send_value], 'Users': [trial.users], 'Points': [points_scored]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

    pprint.pprint(df)
    df.to_csv(join(RESULTS_DIR, 'points_scored.csv'), index=False)

    # Calculate average score for each condition

    # print(df[df.Communication.isin(['Direct'])])z
    # key_names = ['Communication','Send']
    # key_values = ['Direct','Yes']
    # print(df[df[key_names].isin(key_values).all(1)])

    print('Direct-Send mean points:', df[df[['Communication','Condition']].isin(['DIR','RS']).all(1)]['Points'].mean())
    print('Direct-NoSend mean points:', df[df[['Communication','Condition']].isin(['DIR','NRS']).all(1)]['Points'].mean())
    print('Indirect-Send mean points:', df[df[['Communication','Condition']].isin(['IND','RS']).all(1)]['Points'].mean())
    print('Indirect-NoSend mean points:', df[df[['Communication','Condition']].isin(['IND','NRS']).all(1)]['Points'].mean())

    # x_coords = np.array([df.query('Condition=="RS"').Condition, df.query('Condition=="NRS"').Condition])
    # y_coords = np.array([df.query('Condition=="RS"').Points, df.query('Condition=="NRS"').Points])
    # print(x_coords)
    # print(y_coords)

    # # Loop x-coords annd y-coords to create a list of pairs
    # # example: lines = [[(0, 11.44234246), (1, 12.05103481)]]
    # lines = []
    # for i in range(len(x_coords[0])):
    #     pair = [(0, y_coords[0][i]), (1, y_coords[1][i])]
    #     lines.append(pair)

    # fig, axes = plt.subplots(nrows=1, ncols=3)

    # sns.boxplot(data=df, ax=axes[0], x='Communication', y='Points',  palette='Set2', fliersize=0)
    # sns.boxplot(data=df, ax=axes[1], x='Condition', y='Points', order=['RS','NRS'], palette='Set1', fliersize=0)
    # sns.boxplot(data=df, ax=axes[2], x='Condition', y='Points', hue='Communication', order=['RS','NRS'], hue_order=['DIR','IND'], palette='Set2', fliersize=0)

    # sns.stripplot(data=df, ax=axes[0], x='Communication', y='Points', order=['DIR', 'IND'], size=6, color='.25', linewidth=0, dodge=True, jitter=False)
    # sns.stripplot(data=df, ax=axes[1], x='Condition', y='Points', order=['RS','NRS'], size=6, color='.25', linewidth=0, dodge=True, jitter=False)
    # g = sns.stripplot(data=df, ax=axes[2], x='Condition', y='Points',  hue='Communication', order=['RS','NRS'], hue_order=['DIR','IND'], size=6, color='.25', linewidth=0, dodge=True, jitter=False)
    # g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # # statistical annotation
    # pairs = [('DIR', 'IND')]
    # annotator = Annotator(axes[0], pairs, data=df, x='Communication', y='Points', order=['DIR','IND'])
    # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    # pairs = [('RS', 'NRS')]
    # annotator = Annotator(axes[1], pairs, data=df, x='Condition', y='Points', order=['RS', 'NRS'])
    # annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    # # pairs = [
    # #         [('DIR', 'RS'), ('DIR', 'NRS')],
    # #         [('DIR', 'RS'), ('IND', 'RS')],
    # #         [('DIR', 'NRS'), ('IND', 'NRS')],
    # #         [('IND', 'RS'), ('IND', 'NRS')],
    # #         ]
    # # annotator = Annotator(axes[2], pairs, data=df, x='Communication', y='Points', order=['DIR','IND'], hue='Condition')
    # # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    # # annotator.apply_and_annotate()

    # plt.setp(axes, ylim=[55,150])

    # fig.tight_layout()

    # plt.show()

    # Font
    # font = FontProperties()
    # font.set_family('serif')
    # font.set_name('Times New Roman')
    # font.set_size(22)

    # # Plot 1
    # fig = plt.figure(figsize=(5,7))
    # axes = fig.gca()
    # sns.boxplot(data=df, ax=axes, x='Communication', y='Points',  color='skyblue', fliersize=0)
    # sns.stripplot(data=df, ax=axes, x='Communication', y='Points', order=['DIR', 'IND'], size=8, color='.25', linewidth=0, dodge=True, jitter=False)

    # # label
    # plt.rcParams.update({'font.size': 22})

    # axes.set_xlabel('Communication', fontproperties=font)
    # axes.set_ylabel('Points Scored', fontproperties=font)

    # for label in axes.get_xticklabels():
    #     label.set_fontproperties(font)
    # for label in axes.get_yticklabels():
    #     label.set_fontproperties(font)

    # # border
    # axes.spines['top'].set_visible(False)
    # axes.spines['right'].set_visible(False)
    # axes.spines['left'].set_linewidth(1)
    # axes.spines['bottom'].set_linewidth(1)
    # axes.tick_params(width=1)

    # plt.setp(axes, ylim=[55,148])

    # # stats annotation
    # pairs = [('DIR', 'IND')]
    # annotator = Annotator(axes, pairs, data=df, x='Communication', y='Points', order=['DIR','IND'])
    # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    # plt.tight_layout()

    # # plt.show()
    # plt.savefig(join(RESULTS_DIR, 'points_scored_communication.pdf'))

    # # Plot 2
    # fig = plt.figure(figsize=(5,7))
    # axes = fig.gca()
    # sns.boxplot(data=df, ax=axes, x='Condition', y='Points', order=['RS','NRS'], color='skyblue', fliersize=0)
    # sns.stripplot(data=df, ax=axes, x='Condition', y='Points', order=['RS','NRS'], size=8, color='.25', linewidth=0, dodge=True, jitter=False)

    # lc = LineCollection(lines)
    # lc.set_color((0.5, 0.5, 0.5))
    # axes.add_collection(lc)

    # # label
    # plt.rcParams.update({'font.size': 22})

    # axes.set_xlabel('Condition', fontproperties=font)
    # axes.set_ylabel('Points Scored', fontproperties=font)

    # for label in axes.get_xticklabels():
    #     label.set_fontproperties(font)
    # for label in axes.get_yticklabels():
    #     label.set_fontproperties(font)

    # # border
    # axes.spines['top'].set_visible(False)
    # axes.spines['right'].set_visible(False)
    # axes.spines['left'].set_linewidth(1)
    # axes.spines['bottom'].set_linewidth(1)
    # axes.tick_params(width=1)

    # plt.setp(axes, ylim=[55,148])

    # # stats annotation
    # pairs = [('RS', 'NRS')]
    # annotator = Annotator(axes, pairs, data=df, x='Condition', y='Points', order=['RS', 'NRS'])
    # annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    # plt.tight_layout()

    # # plt.show()
    # plt.savefig(join(RESULTS_DIR, 'points_scored_condition.pdf'))
    df_original = df.copy()
    df_overall = df.copy()
    df_overall['Condition'] = 'Overall'
    df = pd.concat([df, df_overall])
    print(df)

    df_rs = df.query('Condition=="RS"')
    df_nrs = df.query('Condition=="NRS"')

    # Font
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_size(18)

    font2 = FontProperties()
    font2.set_family('serif')
    font2.set_name('Times New Roman')
    font2.set_size(14)

    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 5), sharey=True)

    # sns.boxplot(data=df, ax=axes[0], x='Communication', y='Points', order=['DIR','IND'], palette='Set2', fliersize=0)
    # sns.boxplot(data=df_rs, ax=axes[1], x='Communication', y='Points', order=['DIR','IND'], palette='Set2', fliersize=0)
    # sns.boxplot(data=df_nrs, ax=axes[2], x='Communication', y='Points', order=['DIR','IND'], palette='Set2', fliersize=0)

    # sns.stripplot(data=df, ax=axes[0], x='Communication', y='Points', order=['DIR', 'IND'], size=6, color='.25', linewidth=0, dodge=True, jitter=False)
    # sns.stripplot(data=df_rs, ax=axes[1], x='Communication', y='Points', order=['DIR', 'IND'], size=6, color='.25', linewidth=0, dodge=True, jitter=False)
    # sns.stripplot(data=df_nrs, ax=axes[2], x='Communication', y='Points', order=['DIR', 'IND'], size=6, color='.25', linewidth=0, dodge=True, jitter=False)

    # # label
    # axes[0].set_ylabel('Points Scored', fontproperties=font)
    # axes[0].set_xlabel('Overall', fontproperties=font)
    # axes[1].set_xlabel('RS', fontproperties=font)
    # axes[2].set_xlabel('NRS', fontproperties=font)

    # plt.rcParams.update({'font.size': 22})

    # # border
    # for i, ax in enumerate(axes.flat):
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['bottom'].set_linewidth(1)
    #     ax.tick_params(width=1)

    #     if i != 0:
    #         ax.spines['left'].set_visible(False)
    #         ax.axes.yaxis.set_visible(False)

    #     for label in ax.get_xticklabels():
    #         label.set_fontproperties(font)
    #     for label in ax.get_yticklabels():
    #         label.set_fontproperties(font)

    # # statistical annotation
    # pairs = [('DIR', 'IND')]
    # annotator = Annotator(axes[0], pairs, data=df, x='Communication', y='Points', order=['DIR','IND'])
    # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    # annotator = Annotator(axes[1], pairs, data=df_rs, x='Communication', y='Points', order=['DIR','IND'])
    # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    # # annotator.apply_and_annotate()

    # annotator = Annotator(axes[2], pairs, data=df_nrs, x='Communication', y='Points', order=['DIR','IND'])
    # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    # # annotator.apply_and_annotate()

    # # pairs = [
    # #         [('DIR', 'RS'), ('DIR', 'NRS')],
    # #         [('DIR', 'RS'), ('IND', 'RS')],
    # #         [('DIR', 'NRS'), ('IND', 'NRS')],
    # #         [('IND', 'RS'), ('IND', 'NRS')],
    # #         ]
    # # annotator = Annotator(axes[2], pairs, data=df, x='Communication', y='Points', order=['DIR','IND'], hue='Condition')
    # # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    # # annotator.apply_and_annotate()

    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    fig = plt.figure(figsize=(9, 5))
    axes = fig.gca()

    sns.boxplot(data=df, ax=axes, x='Condition', y='Points', order=['Overall', 'RS', 'NRS'], hue='Communication', hue_order=['DIR','IND'], palette='Set2', fliersize=0)

    colors = ['.25', '.25']
    sns.stripplot(data=df, ax=axes, x='Condition', y='Points', order=['Overall', 'RS', 'NRS'], hue='Communication', hue_order=['DIR', 'IND'], size=6, palette=colors, linewidth=0, dodge=True, jitter=False)

    handles, labels = axes.get_legend_handles_labels()
    l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(0.95, 1.05), prop=font2)

    # label
    axes.xaxis.get_label().set_fontproperties(font)
    axes.yaxis.get_label().set_fontproperties(font)

    # border
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_linewidth(1)
    axes.tick_params(width=1)

    for label in axes.get_xticklabels():
        label.set_fontproperties(font)
    for label in axes.get_yticklabels():
        label.set_fontproperties(font)

    start, stop = axes.get_ylim()
    start = math.ceil(start / 10) * 10
    ticks = np.arange(start, stop + 20, 20)
    axes.set_yticks(ticks)

    # axes.set_xlabel('Communication', fontproperties=font)
    axes.set_ylabel('Task score', fontproperties=font)

    plt.rcParams.update({'font.size': 22})

    # statistical annotation
    pairs = [[('Overall', 'DIR'), ('Overall', 'IND')]]
    annotator = Annotator(axes, pairs, data=df, x='Condition', y='Points', order=['Overall', 'RS', 'NRS'], hue='Communication')
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    pairs = [[('RS', 'DIR'), ('RS', 'IND')]]
    annotator = Annotator(axes, pairs, data=df, x='Condition', y='Points', order=['Overall', 'RS', 'NRS'], hue='Communication')
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    pairs = [[('NRS', 'DIR'), ('NRS', 'IND')]]
    annotator = Annotator(axes, pairs, data=df, x='Condition', y='Points', order=['Overall', 'RS', 'NRS'], hue='Communication')
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    # stats annotation
    pairs = [('RS', 'NRS')]
    annotator = Annotator(axes, pairs, data=df_original, x='Condition', y='Points', order=['RS', 'NRS'])
    annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    plt.setp(axes, ylim=[52,155])

    fig.tight_layout()

    # plt.show()
    plt.savefig(join(RESULTS_DIR, 'points_scored_communication.pdf'))


def plot_distance_traveled(stats):
    """Plot the average distance traveled by a robot in each condition (dir_no_send, dir_send, ind_no_send, ind_send)"""
    
    df = pd.DataFrame({
                        'Communication': pd.Series(dtype='str'), 
                        'Condition':     pd.Series(dtype='str'),
                        'Users':         pd.Series(dtype='str'),
                        'Distance':      pd.Series(dtype='float')
                    })

    max_rs = 0
    max_nrs = 0

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:
                print(trial.users, trial.totalPoints)

                # Initialize dict
                robot_dist = {}
                robot_pos = {}

                for robot in trial.data[1]['log'].robots:
                    robot_dist[robot.name] = 0
                    robot_pos[robot.name] = robot.position

                # Loop each timestep to accumulatively count the distance traveled by each robot
                for time, data in trial.data.items():

                    if time == 1: # Skip time = 1
                        continue

                    for robot in data['log'].robots:

                        # Calculate distance traveled from the previous timestep
                        pos = robot.position
                        prev_pos = robot_pos[robot.name]
                        traveled = ((float(pos.x)-float(prev_pos.x))**2)+((float(pos.y)-float(prev_pos.y))**2)**0.5
                        
                        robot_dist[robot.name] += traveled
                        robot_pos[robot.name] = pos

                # Calculate the average distance traveled by a robot
                total_dist = 0
                for robot_name, dist in robot_dist.items():
                    total_dist += dist
                    # Record the maximum distance traveled by a single robot
                    if send == 'send':
                        if dist > max_rs:
                            max_rs = dist
                    elif send == 'no_send':
                        if dist > max_nrs:
                            max_nrs = dist
                average_dist = total_dist / (trial.numLeaders + trial.numWorkers)

                #Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # Add average distance traveled to dataframe
                d = pd.DataFrame({'Communication': [cond_value], 'Condition': [send_value], 'Users': [trial.users], 'Distance': [average_dist]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

                print('average_dist:', average_dist)

    pprint.pprint(df)
    df.to_csv(join(RESULTS_DIR, 'distance_traveled.csv'), index=False)

    print('Maximum distance traveled by a single robot (RS):', max_rs)
    print('Maximum distance traveled by a single robot (NRS):', max_nrs)

    # print(df.query('Condition in ["RS","NRS"]'))
    x_coords = np.array([df.query('Condition=="RS"').Condition, df.query('Condition=="NRS"').Condition])
    y_coords = np.array([df.query('Condition=="RS"').Distance, df.query('Condition=="NRS"').Distance])
    print(x_coords)
    print(y_coords)

    # Loop x-coords annd y-coords to create a list of pairs
    # example: lines = [[(0, 11.44234246), (1, 12.05103481)]]
    lines = []
    for i in range(len(x_coords[0])):
        pair = [(0, y_coords[0][i]), (1, y_coords[1][i])]
        lines.append(pair)

    # Plot average distance traveled by a single robot
    # fig, axes = plt.subplots(nrows=1, ncols=3)

    # # sns.barplot(data=df, ax=axes[0], x='Communication', y='Distance', order=['DIR', 'IND'], palette='Set2')
    # # sns.barplot(data=df, ax=axes[1], x='Condition', y='Distance', order=['RS', 'NRS'], palette='Set1')
    # # g = sns.barplot(data=df, ax=axes[2], x='Communication', y='Distance',  hue='Condition', hue_order=['RS','NRS'], palette='Set1')

    # # g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # sns.boxplot(data=df, ax=axes[0], x='Communication', y='Distance',  palette='Set2', fliersize=0)
    # sns.boxplot(data=df, ax=axes[1], x='Condition', y='Distance', order=['RS','NRS'], palette='Set1', fliersize=0)
    # sns.boxplot(data=df, ax=axes[2], x='Communication', y='Distance', hue='Condition', hue_order=['RS','NRS'], palette='Set1', fliersize=0)

    # sns.stripplot(data=df, ax=axes[0], x='Communication', y='Distance', order=['DIR', 'IND'], size=6, palette='Set2', linewidth=2, dodge=True)
    # sns.stripplot(data=df, ax=axes[1], x='Condition', y='Distance', order=['RS','NRS'], size=6, color='.25', linewidth=0, dodge=True, jitter=False)
    # g = sns.stripplot(data=df, ax=axes[2], x='Communication', y='Distance',  hue='Condition', hue_order=['RS','NRS'], size=6, color='.25', linewidth=0, dodge=True, jitter=False)
    # g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # lc = LineCollection(lines)
    # lc.set_color((0.5, 0.5, 0.5))
    # axes[1].add_collection(lc)

    # # statistical annotation
    # pairs = [('RS', 'NRS')]
    # annotator = Annotator(axes[1], pairs, data=df, x='Condition', y='Distance', order=['RS','NRS'])
    # annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    # plt.setp(axes, ylim=[0,15])

    # # Set y-axis labels
    # for i, ax in enumerate(axes.flat):
    #     ax.set_ylabel("Distance (m)")

    # fig.tight_layout()

    # plt.show()

    # Font
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_size(22)

    # Plot 1
    fig = plt.figure(figsize=(5,7))
    axes = fig.gca()
    sns.boxplot(data=df, ax=axes, x='Condition', y='Distance', order=['RS','NRS'], color='skyblue', fliersize=0)
    sns.stripplot(data=df, ax=axes, x='Condition', y='Distance', order=['RS','NRS'], size=8, color='.25', linewidth=0, dodge=True, jitter=False)

    lc = LineCollection(lines)
    lc.set_color((0.5, 0.5, 0.5))
    axes.add_collection(lc)

    # label
    plt.rcParams.update({'font.size': 22})

    axes.set_xlabel('Condition', fontproperties=font)
    axes.set_ylabel('Distance traveled (m)', fontproperties=font)

    for label in axes.get_xticklabels():
        label.set_fontproperties(font)
    for label in axes.get_yticklabels():
        label.set_fontproperties(font)

    # border
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_linewidth(1)
    axes.spines['bottom'].set_linewidth(1)
    axes.tick_params(width=1)

    plt.setp(axes, ylim=[0,15])

    # stats annotation
    pairs = [('RS', 'NRS')]
    annotator = Annotator(axes, pairs, data=df, x='Condition', y='Distance', order=['RS', 'NRS'])
    annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    plt.tight_layout()

    # plt.show()
    plt.savefig(join(RESULTS_DIR, 'distance_traveled_condition.pdf'))


def plot_robots_shared(stats):
    """Plot the number of robots that were explicitly sent by the leaders in each condition (dir_no_send, dir_send, ind_no_send, ind_send)"""
    
    df = pd.DataFrame({
                        'Communication': pd.Series(dtype='str'), 
                        'Condition':     pd.Series(dtype='str'), 
                        'Robots Sent':   pd.Series(dtype='float')
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:

                # Initialize counter
                total_robots_sent = 0

                # Loop each timestep to accumulatively count the number of robots sent by the leaders
                for time, data in trial.data.items():
                    for command in data['commands']:
                            
                        # Add the number of robots sent by this command
                        if command['type'] == 'send':
                            total_robots_sent += int(command['value'])

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                d = pd.DataFrame({'Communication': [cond_value], 'Condition': [send_value], 'Robots Sent': [total_robots_sent]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

                print('total_robots_sent:', total_robots_sent)

    pprint.pprint(df)

    df_send = df[df['Condition'] == 'RS']
    df_send['Communication'] = df_send['Communication'].replace(['DIR'], 'DIR-RS')
    df_send['Communication'] = df_send['Communication'].replace(['IND'], 'IND-RS')

    pprint.pprint(df_send)
    df_send.to_csv(join(RESULTS_DIR, 'robot_shared.csv'), index=False)

    # Plot average number of connectors
    # fig, axes = plt.subplots(nrows=1, ncols=3)

    # # sns.barplot(data=df_send, ax=axes[0], x='Communication', y='Robots Sent', order=['Direct-Send', 'Indirect-Send'], palette='Set2')
    # # sns.barplot(data=df, ax=axes[1], x='Send', y='Robots Sent', order=['Yes', 'No'], palette='Set1')
    # # sns.barplot(data=df, ax=axes[2], x='Communication', y='Robots Sent',  hue='Send', hue_order=['Yes','No'], palette='Set1')

    # sns.boxplot(data=df_send, ax=axes[0], x='Communication', y='Robots Sent', order=['DIR-RS', 'IND-RS'], palette='Set2')
    # sns.boxplot(data=df, ax=axes[1], x='Condition', y='Robots Sent', order=['RS', 'NRS'], palette='Set1')
    # sns.boxplot(data=df, ax=axes[2], x='Communication', y='Robots Sent',  hue='Condition', hue_order=['RS', 'NRS'], palette='Set1')

    # sns.stripplot(data=df_send, ax=axes[0], x='Communication', y='Robots Sent', order=['DIR-RS', 'IND-RS'], size=6, palette='Set2', linewidth=2, dodge=True)
    # sns.stripplot(data=df, ax=axes[1], x='Condition', y='Robots Sent', order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2)
    # g = sns.stripplot(data=df, ax=axes[2], x='Communication', y='Robots Sent',  hue='Condition', hue_order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    # g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # plt.setp(axes, ylim=[0,60])

    # # statistical annotation
    # pairs = [('DIR-RS', 'IND-RS')]
    # annotator = Annotator(axes[0], pairs, data=df_send, x='Communication', y='Robots Sent', order=['DIR-RS', 'IND-RS'])
    # annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    # fig.tight_layout()

    # plt.show()


    # Font
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_size(22)

    # Plot 1
    fig = plt.figure(figsize=(5,7))
    axes = fig.gca()
    sns.boxplot(data=df_send, ax=axes, x='Communication', y='Robots Sent', order=['DIR-RS','IND-RS'], color='skyblue', fliersize=0)
    sns.stripplot(data=df_send, ax=axes, x='Communication', y='Robots Sent', order=['DIR-RS','IND-RS'], size=8, color='.25', linewidth=0, dodge=True, jitter=False)

    # lc = LineCollection(lines)
    # lc.set_color((0.5, 0.5, 0.5))
    # axes.add_collection(lc)

    # label
    plt.rcParams.update({'font.size': 22})

    axes.set_xlabel('Communication', fontproperties=font)
    axes.set_ylabel('Number of robots shared', fontproperties=font)

    for label in axes.get_xticklabels():
        label.set_fontproperties(font)
    for label in axes.get_yticklabels():
        label.set_fontproperties(font)

    # border
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_linewidth(1)
    axes.spines['bottom'].set_linewidth(1)
    axes.tick_params(width=1)

    plt.setp(axes, ylim=[0,60])

    # stats annotation
    pairs = [('DIR-RS', 'IND-RS')]
    annotator = Annotator(axes, pairs, data=df_send, x='Communication', y='Robots Sent', order=['DIR-RS', 'IND-RS'])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    # annotator.apply_and_annotate()

    plt.tight_layout()

    # plt.show()
    plt.savefig(join(RESULTS_DIR, 'robots_shared.pdf'))


def plot_robots_shared_vs_points(stats, include_partial_points=False):
    """Plot the number of robots that were explicitly sent by the leaders against the points scored"""
    
    df = pd.DataFrame({
                        'Communication':    pd.Series(dtype='str'),
                        'Condition':        pd.Series(dtype='str'),
                        'Points':           pd.Series(dtype='float'),
                        'Robots Sent':      pd.Series(dtype='float')
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:

                # Initialize counter
                total_robots_sent = 0

                # Loop each timestep to accumulatively count the number of robots sent by the leaders
                for time, data in trial.data.items():
                    for command in data['commands']:
                            
                        # Add the number of robots sent by this command
                        if command['type'] == 'send':
                            total_robots_sent += int(command['value'])

                # Count points
                points_scored = trial.totalPoints

                if include_partial_points:

                    partial_points_scored = 0

                    # Check final timestep for any partially completed tasks
                    for task in trial.data[trial.totalTime]['log'].tasks:
                        factor = (300. - 50) / (12. - 1)
                        init_demand = math.floor((task.requiredRobots * factor) + (50 - factor))

                        if task.demand < init_demand:
                            # Calculate points for each unfinished task
                            partial_points_scored += (float(task.demand) / init_demand) * task.requiredRobots
                            # print('Partially completed task of size', task.requiredRobots, '(', task.demand, '/', init_demand, ') Adding', (float(task.demand) / init_demand) * task.requiredRobots)

                    print('Total partial points', partial_points_scored)

                    # Add partially completed tasks to the score
                    points_scored += partial_points_scored

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                d = pd.DataFrame({'Communication': [cond_value], 'Condition': [send_value], 'Points': [points_scored], 'Robots Sent': [total_robots_sent]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

                print('total_robots_sent:', total_robots_sent)

    pprint.pprint(df)

    df_send = df[df['Condition'] == 'RS']
    df_send['Communication'] = df_send['Communication'].replace(['DIR'], 'DIR-RS')
    df_send['Communication'] = df_send['Communication'].replace(['IND'], 'IND-RS')

    pprint.pprint(df_send)

    # Plot points scored against the number of robots sent between leaders
    fig, axes = plt.subplots(nrows=1, ncols=1)

    sns.scatterplot(data=df_send, ax=axes, x='Robots Sent', y='Points', hue='Communication', palette='Set2')

    plt.setp(axes, xlim=[0,50])

    plt.show()


def plot_average_connectors(stats):
    """Plot the average number of connectors between the two teams"""
    
    df = pd.DataFrame({
                        'Communication': pd.Series(dtype='str'), 
                        'Send':          pd.Series(dtype='str'), 
                        'Connectors':    pd.Series(dtype='float')
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:

                # Initialize counter
                total_connectors = 0

                # Loop each timestep to accumulatively count the distance traveled by each robot
                for time, data in trial.data.items():
                    for robot in data['log'].robots:

                        # Add the number of connectors in this timestep
                        if robot.state == time_step_pb2.Robot.CONNECTOR:
                            total_connectors += 1

                # Calculate the average number of connectors
                average_connectors = float(total_connectors) / trial.totalTime

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                d = pd.DataFrame({'Communication': [cond_value], 'Send': [send_value], 'Connectors': [average_connectors]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

                print('average_connectors:', average_connectors)

    pprint.pprint(df)

    # Plot average number of connectors
    fig, axes = plt.subplots(nrows=1, ncols=3)

    sns.barplot(data=df, ax=axes[0], x='Communication', y='Connectors', order=['Direct', 'Indirect'], palette='Set2')
    sns.barplot(data=df, ax=axes[1], x='Send', y='Connectors', order=['Yes', 'No'], palette='Set1')
    sns.barplot(data=df, ax=axes[2], x='Communication', y='Connectors',  hue='Send', hue_order=['Yes','No'], palette='Set1')

    # TODO: line graph with y-axis as time is better?

    plt.show()


def plot_distance_between_teams(stats):
    """Plot the average distance between the two teams"""
    
    df = pd.DataFrame({
                        'Communication': pd.Series(dtype='str'), 
                        'Condition':     pd.Series(dtype='str'), 
                        'Users':         pd.Series(dtype='str'),
                        'Distance':      pd.Series(dtype='float')
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:

                # Initialize list
                team_dist = []

                # For each tiemstep
                for time, data in trial.data.items():
                    # Add robot positions into separate arrays
                    team_pos = {}
                    team_robot_pos = {}
                    for team in trial.teams:
                        team_robot_pos[team] = []

                    # Add robot positions according to their teams
                    for robot in data['log'].robots:
                        for team in trial.teams:
                            if robot.teamID == team:
                                team_robot_pos[team].append([robot.position.x, robot.position.y])

                    # Calculate the average team positions
                    for key, value in team_robot_pos.items():
                        team_pos[key] = np.average(np.array(value), axis=0)

                    # Calculate the distance between the two teams
                    x1 = team_pos[1][0] # team 1
                    y1 = team_pos[1][1]
                    x2 = team_pos[2][0] # team 2
                    y2 = team_pos[2][1]
                    distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

                    team_dist.append(distance)

                average_dist = np.average(np.array(team_dist), axis=0)

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # Add average distance traveled to dataframe
                d = pd.DataFrame({'Communication': [cond_value], 'Condition': [send_value], 'Users': [trial.users], 'Distance': [average_dist]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

                print('average_dist:', average_dist)

    pprint.pprint(df)
    df.to_csv(join(RESULTS_DIR, 'team_distance.csv'), index=False)

    # Plot average distance traveled by a single robot
    fig, axes = plt.subplots(nrows=1, ncols=3)

    sns.barplot(data=df, ax=axes[0], x='Communication', y='Distance', order=['DIR', 'IND'], palette='Set2')
    sns.barplot(data=df, ax=axes[1], x='Condition', y='Distance', order=['RS', 'NRS'], palette='Set1')
    g = sns.barplot(data=df, ax=axes[2], x='Communication', y='Distance',  hue='Condition', hue_order=['RS','NRS'], palette='Set1')
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # Set y-axis labels
    for i, ax in enumerate(axes.flat):
        ax.set_ylabel("Distance (m)")

    plt.setp(axes, ylim=[0,1.7])

    fig.tight_layout()

    plt.show()

    # TODO: 'send' resulted in longer distance between teams. Does 'Send' allow teams to work further away? What is the performance of teams in 'no_send'? Is it low?


def plot_distance_between_teams_timeline(stats):
    """Plot the average distance between the two teams in each timestep"""

    df = pd.DataFrame({
                        'Trial':            pd.Series(dtype='str'),
                        'Time':             pd.Series(dtype='int'),
                        'Communication':    pd.Series(dtype='str'), 
                        'Send':             pd.Series(dtype='str'), 
                        'Distance':         pd.Series(dtype='float')
                    })

    trial_id = 0

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:
                print(trial.users, trial.totalPoints)

                trial_id += 1

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # For each tiemstep
                for time, data in trial.data.items():
                    if int(time) == 1 or int(time) % 100 == 0:
                        # Add robot positions into separate arrays
                        team_pos = {}
                        team_robot_pos = {}
                        for team in trial.teams:
                            team_robot_pos[team] = []

                        # Add robot positions according to their teams
                        for robot in data['log'].robots:
                            for team in trial.teams:
                                if robot.teamID == team:
                                    team_robot_pos[team].append([robot.position.x, robot.position.y])

                        # Calculate the average team positions
                        for key, value in team_robot_pos.items():
                            team_pos[key] = np.average(np.array(value), axis=0)

                        # Calculate the distance between the two teams
                        x1 = team_pos[1][0] # team 1
                        y1 = team_pos[1][1]
                        x2 = team_pos[2][0] # team 2
                        y2 = team_pos[2][1]
                        distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

                        # Add team distance for this timestep to dataframe
                        d = pd.DataFrame({'Trial': ['trial' + str(trial_id)], 'Time': [time/10.], 'Communication': [cond_value], 'Send': [send_value], 'Distance': [distance]})
                        df = pd.concat([df, d], ignore_index=True, axis=0)

                # break
            # break
        # break

    pprint.pprint(df)

    # Font
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_size(18)

    font2 = FontProperties()
    font2.set_family('serif')
    font2.set_name('Times New Roman')
    font2.set_size(14)

    # Plot distance between teams over the mission
    # fig, axes = plt.subplots(nrows=1, ncols=1)

    fig = plt.figure(figsize=(9,4))
    axes = fig.gca()

    # sns.scatterplot(data=df, ax=axes, x='Average Taskload', y='Points', hue='Communication', hue_order=['Direct', 'Indirect'], style='Communication')

    sns.lineplot(data=df, ax=axes, x='Time', y='Distance', hue='Send', hue_order=['RS', 'NRS'], palette='Set1')

    plt.yticks(np.arange(0.5, 2.5, 0.5))
    plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc='lower center', ncol=2, frameon=False, prop=font2)

    axes.set_xlabel('Time (s)', fontproperties=font)
    axes.set_ylabel('Team separation (m)', fontproperties=font)

    for label in axes.get_xticklabels():
        label.set_fontproperties(font)
    for label in axes.get_yticklabels():
        label.set_fontproperties(font)

    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_linewidth(1)
    axes.spines['bottom'].set_linewidth(1)
    axes.tick_params(width=1)

    plt.xlim([0,600])

    fig.tight_layout()

    # plt.show()
    plt.savefig(join(RESULTS_DIR, 'team_separation_condition.pdf'))


def plot_distance_between_teams_vs_points(stats, include_partial_points=False):
    """Plot the average distance between the two teams against points scored"""
    
    df = pd.DataFrame({
                        'Communication':    pd.Series(dtype='str'), 
                        'Send':             pd.Series(dtype='str'), 
                        'Points':           pd.Series(dtype='float'),
                        'Distance':         pd.Series(dtype='float')
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:

                # Initialize list
                team_dist = []

                # For each tiemstep
                for time, data in trial.data.items():
                    # Add robot positions into separate arrays
                    team_pos = {}
                    team_robot_pos = {}
                    for team in trial.teams:
                        team_robot_pos[team] = []

                    # Add robot positions according to their teams
                    for robot in data['log'].robots:
                        for team in trial.teams:
                            if robot.teamID == team:
                                team_robot_pos[team].append([robot.position.x, robot.position.y])

                    # Calculate the average team positions
                    for key, value in team_robot_pos.items():
                        team_pos[key] = np.average(np.array(value), axis=0)

                    # Calculate the distance between the two teams
                    x1 = team_pos[1][0] # team 1
                    y1 = team_pos[1][1]
                    x2 = team_pos[2][0] # team 2
                    y2 = team_pos[2][1]
                    distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5

                    team_dist.append(distance)

                average_dist = np.average(np.array(team_dist), axis=0)

                # Count points
                points_scored = trial.totalPoints

                if include_partial_points:

                    partial_points_scored = 0

                    # Check final timestep for any partially completed tasks
                    for task in trial.data[trial.totalTime]['log'].tasks:
                        factor = (300. - 50) / (12. - 1)
                        init_demand = math.floor((task.requiredRobots * factor) + (50 - factor))

                        if task.demand < init_demand:
                            # Calculate points for each unfinished task
                            partial_points_scored += (float(task.demand) / init_demand) * task.requiredRobots
                            # print('Partially completed task of size', task.requiredRobots, '(', task.demand, '/', init_demand, ') Adding', (float(task.demand) / init_demand) * task.requiredRobots)

                    print('Total partial points', partial_points_scored)

                    # Add partially completed tasks to the score
                    points_scored += partial_points_scored

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # Add average distance traveled to dataframe
                d = pd.DataFrame({'Communication': [cond_value], 'Send': [send_value], 'Points': [points_scored], 'Distance': [average_dist]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

                print('average_dist:', average_dist)

    pprint.pprint(df)

    # Plot points scored against the number of robots sent between leaders
    fig, axes = plt.subplots(nrows=1, ncols=1)

    sns.scatterplot(data=df, ax=axes, x='Distance', y='Points', hue='Communication', style='Send')

    # Set y-axis labels
    # for i, ax in enumerate(axes.flat):
    #     ax.set_xlabel("Distance (m)")
    axes.set_xlabel("Distance (m)")

    # plt.setp(axes, ylim=[0,1.7])

    plt.show()


def plot_traveler_time(stats):
    """Plot the total time a robot spent traveling between the two teams against points scored"""

    df = pd.DataFrame({
                        'Communication':    pd.Series(dtype='str'), 
                        'Condition':        pd.Series(dtype='str'), 
                        'Users':            pd.Series(dtype='str'),
                        'Time':             pd.Series(dtype='float')
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:
                print(trial.users, trial.totalPoints)

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # Total time spent as traveler
                # Loop each timestep and add the number of travelers
                # For each tiemstep

                total_traveler_time = 0

                for time, data in trial.data.items():

                    # Add robot positions according to their teams
                    for robot in data['log'].robots:
                        if robot.state == time_step_pb2.Robot.TRAVELER:
                            total_traveler_time += 1

                total_traveler_time /= 10. # Divide by 10 to convert from simulation time to real time

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # Add average distance traveled to dataframe
                d = pd.DataFrame({'Communication': [cond_value], 'Condition': [send_value], 'Users': [trial.users], 'Time': [total_traveler_time]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

                print('total_traveler_time:', total_traveler_time)

    pprint.pprint(df)
    df_send = df[df['Condition'] == 'RS']
    pprint.pprint(df_send)

    df_send['Communication'] = df_send['Communication'].replace(['DIR'], 'DIR-RS')
    df_send['Communication'] = df_send['Communication'].replace(['IND'], 'IND-RS')

    df_send.to_csv(join(RESULTS_DIR, 'traveler_time.csv'), index=False)

    # Plot points scored against the number of robots sent between leaders
    fig, axes = plt.subplots(nrows=1, ncols=1)

    # sns.scatterplot(data=df_send, ax=axes, x='Time', y='Points', hue='Communication', style='Communication')

    sns.boxplot(data=df_send, ax=axes, x='Communication', y='Time', order=['DIR-RS', 'IND-RS'], palette='Set2')

    sns.stripplot(data=df_send, ax=axes, x='Communication', y='Time', order=['DIR-RS', 'IND-RS'], size=6, palette='Set2', linewidth=2, dodge=True)

    plt.setp(axes, ylim=[0,730])

    plt.show()


def plot_traveler_time_vs_points(stats, include_partial_points=False):
    """Plot the total time a robot spent traveling between the two teams against points scored"""

    df = pd.DataFrame({
                        'Communication':    pd.Series(dtype='str'), 
                        'Condition':        pd.Series(dtype='str'), 
                        'Points':           pd.Series(dtype='float'),
                        'Time':             pd.Series(dtype='float')
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:
                print(trial.users, trial.totalPoints)

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # Points scored
                points_scored = trial.totalPoints

                if include_partial_points:

                    partial_points_scored = 0

                    # Check final timestep for any partially completed tasks
                    for task in trial.data[trial.totalTime]['log'].tasks:
                        factor = (300. - 50) / (12. - 1)
                        init_demand = math.floor((task.requiredRobots * factor) + (50 - factor))

                        if task.demand < init_demand:
                            # Calculate points for each unfinished task
                            partial_points_scored += (float(task.demand) / init_demand) * task.requiredRobots
                            # print('Partially completed task of size', task.requiredRobots, '(', task.demand, '/', init_demand, ') Adding', (float(task.demand) / init_demand) * task.requiredRobots)

                    print('Total partial points', partial_points_scored)

                    # Add partially completed tasks to the score
                    points_scored += partial_points_scored

                # Total time spent as traveler
                # Loop each timestep and add the number of travelers
                # For each tiemstep

                total_traveler_time = 0

                for time, data in trial.data.items():

                    # Add robot positions according to their teams
                    for robot in data['log'].robots:
                        if robot.state == time_step_pb2.Robot.TRAVELER:
                            total_traveler_time += 1

                total_traveler_time /= 10. # Divide by 10 to convert from simulation time to real time

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # Add average distance traveled to dataframe
                d = pd.DataFrame({'Communication': [cond_value], 'Condition': [send_value], 'Points': [points_scored], 'Time': [total_traveler_time]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

                print('total_traveler_time:', total_traveler_time)

    pprint.pprint(df)
    df_send = df[df['Condition'] == 'RS']
    pprint.pprint(df_send)

    # Plot points scored against the number of robots sent between leaders
    fig, axes = plt.subplots(nrows=1, ncols=1)

    sns.scatterplot(data=df_send, ax=axes, x='Time', y='Points', hue='Communication', palette='Set2')
    g = sns.regplot(data=df_send, ax=axes, x='Time', y='Points', scatter_kws={'s':0}, line_kws={'color':'purple'})
    # g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=1)
    # sns.boxplot(data=df_send, ax=axes, x='Time', y='Points', hue='Communication', hue_order=['Diirect','Indirect'], palette='Set2')

    # plt.setp(axes, xlim=[0,50])

    plt.tight_layout()

    plt.show()


def plot_traveler_distance(stats):
    """Plot the distance a robot spent traveling between the two teams"""

    df = pd.DataFrame({
                        'Communication':    pd.Series(dtype='str'), 
                        'Condition':        pd.Series(dtype='str'), 
                        'Users':            pd.Series(dtype='str'),
                        'Distance':         pd.Series(dtype='float')
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:
                print(trial.users, trial.totalPoints)

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # Initialize dict
                robot_dist = {}
                robot_pos = {}

                for robot in trial.data[1]['log'].robots:
                    robot_dist[robot.name] = 0
                    robot_pos[robot.name] = None

                # Loop each timestep to accumulatively count the distance traveled by each robot
                for time, data in trial.data.items():

                    if time == 1: # Skip time = 1
                        continue

                    for robot in data['log'].robots:
                        if robot.state == time_step_pb2.Robot.TRAVELER:

                            if robot_pos[robot.name]:
                                # Calculate distance traveled from the previous timestep
                                pos = robot.position
                                prev_pos = robot_pos[robot.name]
                                traveled = ((float(pos.x)-float(prev_pos.x))**2)+((float(pos.y)-float(prev_pos.y))**2)**0.5
                                
                                robot_dist[robot.name] += traveled
                                robot_pos[robot.name] = pos

                            else:
                                # If robot has turned into a traveler, store current position
                                robot_pos[robot.name] = robot.position

                        else:
                            if robot_pos[robot.name]:
                                # If robot is no longer a traveler, reset its current position
                                robot_pos[robot.name] = None

                # Calculate the average distance traveled by  robot
                total_traveler_dist = 0
                for robot_name, dist in robot_dist.items():
                    total_traveler_dist += dist
                # average_dist = total_dist / (trial.numLeaders + trial.numWorkers)

                # Add average distance traveled to dataframe
                d = pd.DataFrame({'Communication': [cond_value], 'Condition': [send_value], 'Users': [trial.users], 'Distance': [total_traveler_dist]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

                print('total_traveler_dist:', total_traveler_dist)

    pprint.pprint(df)
    df_send = df[df['Condition'] == 'RS']
    pprint.pprint(df_send)

    df_send['Communication'] = df_send['Communication'].replace(['DIR'], 'DIR-RS')
    df_send['Communication'] = df_send['Communication'].replace(['IND'], 'IND-RS')

    df_send.to_csv(join(RESULTS_DIR, 'traveler_dist.csv'), index=False)

    # Plot total distance traveled by the travelers
    fig, axes = plt.subplots(nrows=1, ncols=1)

    sns.boxplot(data=df_send, ax=axes, x='Communication', y='Distance', order=['DIR-RS', 'IND-RS'], palette='Set2', fliersize=0)

    sns.stripplot(data=df_send, ax=axes, x='Communication', y='Distance', order=['DIR-RS', 'IND-RS'], size=6, color=".3", linewidth=0, dodge=True, jitter=False)

    # statistical annotation
    pairs = [('DIR-RS', 'IND-RS')]
    annotator = Annotator(axes, pairs, data=df_send, x='Communication', y='Distance', order=['DIR-RS', 'IND-RS'])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    # plt.setp(axes, ylim=[0,730])

    plt.show()


def plot_average_traveler_distance(stats):
    """Plot the distance a robot spent traveling between the two teams"""

    df = pd.DataFrame({
                        'Communication':    pd.Series(dtype='str'), 
                        'Condition':        pd.Series(dtype='str'), 
                        'Users':            pd.Series(dtype='str'),
                        'Distance':         pd.Series(dtype='float')
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:
                print(trial.users, trial.totalPoints)

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # Initialize dict
                robot_dist = {}
                robot_pos = {}
                traveler_robots = set()

                for robot in trial.data[1]['log'].robots:
                    robot_dist[robot.name] = 0
                    robot_pos[robot.name] = None

                # Loop each timestep to accumulatively count the distance traveled by each robot
                for time, data in trial.data.items():

                    if time == 1: # Skip time = 1
                        continue

                    for robot in data['log'].robots:
                        if robot.state == time_step_pb2.Robot.TRAVELER:

                            traveler_robots.add(robot.name)

                            if robot_pos[robot.name]:
                                # Calculate distance traveled from the previous timestep
                                pos = robot.position
                                prev_pos = robot_pos[robot.name]
                                traveled = ((float(pos.x)-float(prev_pos.x))**2)+((float(pos.y)-float(prev_pos.y))**2)**0.5
                                
                                robot_dist[robot.name] += traveled
                                robot_pos[robot.name] = pos

                            else:
                                # If robot has turned into a traveler, store current position
                                robot_pos[robot.name] = robot.position

                        else:
                            if robot_pos[robot.name]:
                                # If robot is no longer a traveler, reset its current position
                                robot_pos[robot.name] = None

                # Calculate the average distance traveled by  robot
                total_traveler_dist = 0
                average_traveler_dist = 0
                
                for robot_name, dist in robot_dist.items():
                    total_traveler_dist += dist

                if total_traveler_dist:
                    average_traveler_dist = total_traveler_dist / len(traveler_robots)

                # Add average distance traveled to dataframe
                d = pd.DataFrame({'Communication': [cond_value], 'Condition': [send_value], 'Users': [trial.users], 'Distance': [average_traveler_dist]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

                print('average_traveler_dist:', average_traveler_dist)

    pprint.pprint(df)
    df_send = df[df['Condition'] == 'RS']
    pprint.pprint(df_send)

    df_send['Communication'] = df_send['Communication'].replace(['DIR'], 'DIR-RS')
    df_send['Communication'] = df_send['Communication'].replace(['IND'], 'IND-RS')

    df_send.to_csv(join(RESULTS_DIR, 'average_traveler_dist.csv'), index=False)

    # Plot total distance traveled by the travelers
    fig, axes = plt.subplots(nrows=1, ncols=1)

    sns.boxplot(data=df_send, ax=axes, x='Communication', y='Distance', order=['DIR-RS', 'IND-RS'], palette='Set2')

    sns.stripplot(data=df_send, ax=axes, x='Communication', y='Distance', order=['DIR-RS', 'IND-RS'], size=6, palette='Set2', linewidth=2, dodge=True)

    # plt.setp(axes, ylim=[0,730])

    plt.show()


def plot_traveler_distance_vs_points(stats, include_partial_points=False):
    """Plot the distance the robots spent traveling between the two teams against points scored"""

    df = pd.DataFrame({
                        'Communication':    pd.Series(dtype='str'), 
                        'Condition':        pd.Series(dtype='str'), 
                        'Users':            pd.Series(dtype='str'),
                        'Distance':         pd.Series(dtype='float'),
                        'Points':           pd.Series(dtype='float'),
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:
                print(trial.users, trial.totalPoints)

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # Initialize dict
                robot_dist = {}
                robot_pos = {}

                for robot in trial.data[1]['log'].robots:
                    robot_dist[robot.name] = 0
                    robot_pos[robot.name] = None

                # Loop each timestep to accumulatively count the distance traveled by each robot
                for time, data in trial.data.items():

                    if time == 1: # Skip time = 1
                        continue

                    for robot in data['log'].robots:
                        if robot.state == time_step_pb2.Robot.TRAVELER:

                            if robot_pos[robot.name]:
                                # Calculate distance traveled from the previous timestep
                                pos = robot.position
                                prev_pos = robot_pos[robot.name]
                                traveled = ((float(pos.x)-float(prev_pos.x))**2)+((float(pos.y)-float(prev_pos.y))**2)**0.5
                                
                                robot_dist[robot.name] += traveled
                                robot_pos[robot.name] = pos

                            else:
                                # If robot has turned into a traveler, store current position
                                robot_pos[robot.name] = robot.position

                        else:
                            if robot_pos[robot.name]:
                                # If robot is no longer a traveler, reset its current position
                                robot_pos[robot.name] = None

                # Calculate the average distance traveled by  robot
                total_traveler_dist = 0
                for robot_name, dist in robot_dist.items():
                    total_traveler_dist += dist
                # average_dist = total_dist / (trial.numLeaders + trial.numWorkers)

                # Points scored
                points_scored = trial.totalPoints

                if include_partial_points:

                    partial_points_scored = 0

                    # Check final timestep for any partially completed tasks
                    for task in trial.data[trial.totalTime]['log'].tasks:
                        factor = (300. - 50) / (12. - 1)
                        init_demand = math.floor((task.requiredRobots * factor) + (50 - factor))

                        if task.demand < init_demand:
                            # Calculate points for each unfinished task
                            partial_points_scored += (float(task.demand) / init_demand) * task.requiredRobots
                            # print('Partially completed task of size', task.requiredRobots, '(', task.demand, '/', init_demand, ') Adding', (float(task.demand) / init_demand) * task.requiredRobots)

                    print('Total partial points', partial_points_scored)

                    # Add partially completed tasks to the score
                    points_scored += partial_points_scored

                # Add average distance traveled to dataframe
                d = pd.DataFrame({'Communication': [cond_value], 'Condition': [send_value], 'Users': [trial.users], 'Distance': [total_traveler_dist], 'Points': [points_scored]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

                print('total_traveler_dist:', total_traveler_dist)

    pprint.pprint(df)
    df_send = df[df['Condition'] == 'RS']
    pprint.pprint(df_send)

    df_send['Communication'] = df_send['Communication'].replace(['DIR'], 'DIR-RS')
    df_send['Communication'] = df_send['Communication'].replace(['IND'], 'IND-RS')

    df_send.to_csv(join(RESULTS_DIR, 'traveler_dist.csv'), index=False)

    # Plot total distance traveled by the travelers
    fig, axes = plt.subplots(nrows=1, ncols=1)

    # sns.boxplot(data=df_send, ax=axes, x='Communication', y='Distance', order=['DIR-RS', 'IND-RS'], palette='Set2')

    # sns.stripplot(data=df_send, ax=axes, x='Communication', y='Distance', order=['DIR-RS', 'IND-RS'], size=6, palette='Set2', linewidth=2, dodge=True)

    sns.scatterplot(data=df_send, ax=axes, x='Distance', y='Points', hue='Communication', style='Communication')

    # plt.setp(axes, ylim=[0,730])

    plt.show()


def plot_task_waiting_time_vs_points(stats, include_partial_points=False):
    """Plot the total time a leader was in the task, but the task demand did not decrease with respect to the points scored"""

    # Need to keep track of whether a leader is inside a task area
    # Position in log. Add task dimensions
    # If at least one leader is inside task area, keep track of the task name and demand
    # If leader is inside and demand hasn't changed, add it to total time

    # Refer to traveler distance
    # Task appears while reading log

    # Task dimensions {demand: length}
    task_dimensions = {
                        1: 0.4,
                        3: 0.6,
                        6: 0.6,
                        9: 0.8,
                        12: 1.0
                    }

    df = pd.DataFrame({
                        'Communication':    pd.Series(dtype='str'), 
                        'Condition':        pd.Series(dtype='str'), 
                        'Users':            pd.Series(dtype='str'),
                        'Time':             pd.Series(dtype='float'),
                        'Average Time':     pd.Series(dtype='float'),
                        'Points':           pd.Series(dtype='float'),
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:
                print(trial.users, trial.totalPoints)

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)

                # Initialize dict
                task_waiting_time = {}

                for task in trial.data[1]['log'].tasks:
                    task_waiting_time[task.name] = 0

                # Loop each timestep to accumulatively count the distance traveled by each robot
                for time, data in trial.data.items():

                    if time == 1: # Skip time = 1
                        continue

                    for task in data['log'].tasks:
                        
                        task_waiting = False

                        # If task is not in task_pos, create entry
                        if task.name not in task_waiting_time:
                            task_waiting_time[task.name] = 0

                        # Check if a leader is inside the task area
                        for robot in data['log'].robots:
                            if robot.state == time_step_pb2.Robot.LEADER:
                                task_length = task_dimensions[task.requiredRobots]
                                task_x1 = task.position.x + 0.5 * task_length
                                task_x2 = task.position.x - 0.5 * task_length
                                task_y1 = task.position.y + 0.5 * task_length
                                task_y2 = task.position.y - 0.5 * task_length

                                if robot.position.x < task_x1 and robot.position.x > task_x2 and \
                                   robot.position.y < task_y1 and robot.position.y > task_y2:

                                    if task.currentRobots < task.requiredRobots:
                                        task_waiting = True
                                        break

                        if task_waiting:
                            task_waiting_time[task.name] += 1

                # Calculate total waiting time
                total_waiting_time = 0
                for task_name,  time in task_waiting_time.items():
                    total_waiting_time += time
                
                total_waiting_time /= 10. # Divide by 10 to convert from simulation time to real time
                print('total_waiting_time:', total_waiting_time)

                total_tasks = len(task_waiting_time)
                average_waiting_time = total_waiting_time / total_tasks
                print('average_waiting_time:', average_waiting_time)

                # Points scored
                points_scored = trial.totalPoints

                if include_partial_points:

                    partial_points_scored = 0

                    # Check final timestep for any partially completed tasks
                    for task in trial.data[trial.totalTime]['log'].tasks:
                        factor = (300. - 50) / (12. - 1)
                        init_demand = math.floor((task.requiredRobots * factor) + (50 - factor))

                        if task.demand < init_demand:
                            # Calculate points for each unfinished task
                            partial_points_scored += (float(task.demand) / init_demand) * task.requiredRobots
                            # print('Partially completed task of size', task.requiredRobots, '(', task.demand, '/', init_demand, ') Adding', (float(task.demand) / init_demand) * task.requiredRobots)

                    print('Total partial points', partial_points_scored)

                    # Add partially completed tasks to the score
                    points_scored += partial_points_scored

                # Add average distance traveled to dataframe
                d = pd.DataFrame({'Communication': [cond_value], 'Condition': [send_value], 'Users': [trial.users], 'Time': [total_waiting_time], 'Average Time': [average_waiting_time], 'Points': [points_scored]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

    pprint.pprint(df)
    df.to_csv(join(RESULTS_DIR, 'waiting_time.csv'), index=False)

    # Plot total distance traveled by the travelers
    # fig, axes = plt.subplots(nrows=1, ncols=3)

    # sns.boxplot(data=df, ax=axes[0], x='Communication', y='Time',  palette='Set2')
    # sns.boxplot(data=df, ax=axes[1], x='Condition', y='Time', order=['RS','NRS'], palette='Set1')
    # sns.boxplot(data=df, ax=axes[2], x='Communication', y='Time', hue='Condition', hue_order=['RS','NRS'], palette='Set1')

    # sns.stripplot(data=df, ax=axes[0], x='Communication', y='Time', order=['DIR', 'IND'], size=6, palette='Set2', linewidth=2, dodge=True)
    # sns.stripplot(data=df, ax=axes[1], x='Condition', y='Time', order=['RS','NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    # g = sns.stripplot(data=df, ax=axes[2], x='Communication', y='Time',  hue='Condition', hue_order=['RS','NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    # g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # # Set y-axis labels
    # for i, ax in enumerate(axes.flat):
    #     ax.set_ylabel("Waiting time (s)")

    # plt.setp(axes, ylim=[0,700])
    # fig.tight_layout()

    # fig2, axes2 = plt.subplots(nrows=1, ncols=3)

    # sns.boxplot(data=df, ax=axes2[0], x='Communication', y='Average Time',  palette='Set2')
    # sns.boxplot(data=df, ax=axes2[1], x='Condition', y='Average Time', order=['RS','NRS'], palette='Set1')
    # sns.boxplot(data=df, ax=axes2[2], x='Communication', y='Average Time', hue='Condition', hue_order=['RS','NRS'], palette='Set1')

    # sns.stripplot(data=df, ax=axes2[0], x='Communication', y='Average Time', order=['DIR', 'IND'], size=6, palette='Set2', linewidth=2, dodge=True)
    # sns.stripplot(data=df, ax=axes2[1], x='Condition', y='Average Time', order=['RS','NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    # g = sns.stripplot(data=df, ax=axes2[2], x='Communication', y='Average Time',  hue='Condition', hue_order=['RS','NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    # g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # # Set y-axis labels
    # for i, ax in enumerate(axes2.flat):
    #     ax.set_ylabel("Average waiting time per task (s)")

    # plt.setp(axes2, ylim=[0,50])

    # fig2.tight_layout()

    # fig3, axes3 = plt.subplots(nrows=1, ncols=1)

    # sns.scatterplot(data=df, ax=axes3, x='Time', y='Points', hue='Condition', hue_order=['RS', 'NRS'], style='Communication', palette='Set1')

    # # Set y-axis labels
    # axes3.set_xlabel("Waiting time (s)")

    # fig3.tight_layout()

    sns.set_style("darkgrid")

    # Font
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')
    font.set_size(16)

    font2 = FontProperties()
    font2.set_family('serif')
    font2.set_name('Times New Roman')
    font2.set_size(12)

    fig4 = plt.figure(figsize=(7, 5))
    axes4 = fig4.gca()

    # sns.set_theme()
    # sns.set_style('darkgrid', {'axes.grid' : True})

    sns.scatterplot(data=df, ax=axes4, x='Average Time', y='Points', hue='Condition', hue_order=['RS', 'NRS'], style='Communication', palette='Set1')
    g = sns.regplot(data=df, ax=axes4, x='Average Time', y='Points', scatter_kws={'s':0}, line_kws={'color':'purple'})
    g.legend(loc='center left', bbox_to_anchor=(0.68, 0.8), ncol=1, prop=font2)

    # Set axis labels
    axes4.set_xlabel("Average waiting time per task (s)")
    axes4.set_ylabel("Task performance (points)")

    # label
    axes4.xaxis.get_label().set_fontproperties(font)
    axes4.yaxis.get_label().set_fontproperties(font)

    # border
    # axes4.spines['top'].set_visible(False)
    # axes4.spines['right'].set_visible(False)
    # axes4.spines['bottom'].set_linewidth(1)
    # axes4.tick_params(width=1)

    for label in axes4.get_xticklabels():
        label.set_fontproperties(font)
    for label in axes4.get_yticklabels():
        label.set_fontproperties(font)

    plt.setp(axes4, xlim=[4,45])
    plt.setp(axes4, ylim=[52,140])

    fig4.tight_layout()

    # plt.show()
    plt.savefig(join(RESULTS_DIR, 'waiting_time_vs_points.pdf'))


def plot_learning_effect(stats, include_partial_points=False):
    """Plot the learning effect with respect to the points scored"""
    
    df = pd.DataFrame({
                        'Communication':    pd.Series(dtype='str'),
                        'Order':            pd.Series(dtype='str'), 
                        'Group':            pd.Series(dtype='str'),
                        'Points':           pd.Series(dtype='float'), 
                    })

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:

                points_scored = trial.totalPoints

                if include_partial_points:

                    partial_points_scored = 0

                    # Check final timestep for any partially completed tasks
                    for task in trial.data[trial.totalTime]['log'].tasks:
                        factor = (300. - 50) / (12. - 1)
                        init_demand = math.floor((task.requiredRobots * factor) + (50 - factor))

                        if task.demand < init_demand:
                            # Calculate points for each unfinished task
                            partial_points_scored += (float(task.demand) / init_demand) * task.requiredRobots
                            # print('Partially completed task of size', task.requiredRobots, '(', task.demand, '/', init_demand, ') Adding', (float(task.demand) / init_demand) * task.requiredRobots)

                    print('Total partial points', partial_points_scored)

                    # Add partially completed tasks to the score
                    points_scored += partial_points_scored

                order_label = ''
                group_label = ''

                if int(cond[-1]) == 1:
                    if send == 'no_send':
                        order_label = 'Trial 1'
                    else:
                        order_label = 'Trial 2'
                else:
                    if send == 'no_send':
                        order_label = 'Trial 2'
                    else:
                        order_label = 'Trial 1'

                cond_value, send_value = get_config(cond, send)

                d = pd.DataFrame({'Communication': [cond_value], 'Order': [order_label], 'Group': [cond], 'Points': [points_scored]})
                df = pd.concat([df, d], ignore_index=True, axis=0)

    pprint.pprint(df)

    df_dir1 = df[df['Group'] == 'dir1']
    df_dir2 = df[df['Group'] == 'dir2']
    df_ind1 = df[df['Group'] == 'ind1']
    df_ind2 = df[df['Group'] == 'ind2']

    # Plot average distance traveled by a single robot
    fig, axes = plt.subplots(nrows=2, ncols=3)

    sns.boxplot(data=df, ax=axes[0,0], x='Order', y='Points', order=['Trial 1', 'Trial 2'], palette='Set2', fliersize=0)
    sns.boxplot(data=df_dir1, ax=axes[0,1], x='Order', y='Points', order=['Trial 1', 'Trial 2'], palette='Set2', fliersize=0)
    sns.boxplot(data=df_dir2, ax=axes[0,2], x='Order', y='Points', order=['Trial 1', 'Trial 2'], palette='Set2', fliersize=0)
    sns.boxplot(data=df_ind1, ax=axes[1,1], x='Order', y='Points', order=['Trial 1', 'Trial 2'], palette='Set2', fliersize=0)
    sns.boxplot(data=df_ind2, ax=axes[1,2], x='Order', y='Points', order=['Trial 1', 'Trial 2'], palette='Set2', fliersize=0)

    sns.stripplot(data=df, ax=axes[0,0], x='Order', y='Points', order=['Trial 1', 'Trial 2'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df_dir1, ax=axes[0,1], x='Order', y='Points', order=['Trial 1', 'Trial 2'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df_dir2, ax=axes[0,2], x='Order', y='Points', order=['Trial 1', 'Trial 2'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df_ind1, ax=axes[1,1], x='Order', y='Points', order=['Trial 1', 'Trial 2'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df_ind2, ax=axes[1,2], x='Order', y='Points', order=['Trial 1', 'Trial 2'], size=6, palette='Set2', linewidth=2, dodge=True)

    # Turn off axis for unused subplot
    axes[1,0].axis('off')

    # Set y-axis labels
    # for i, ax in enumerate(axes.flat):
    #     ax.set_ylabel("Distance (m)")

    # Set plot titles
    axes[0,0].set_title('Overall')
    axes[0,1].set_title('DIR (NRS -> RS)')
    axes[0,2].set_title('DIR (RS -> NRS)')
    axes[1,1].set_title('IND (NRS -> RS)')
    axes[1,2].set_title('IND (RS -> NRS)')

    plt.setp(axes, ylim=[70,140])

    fig.tight_layout()

    plt.show()


def plot_preliminary(df):
    """Plot the preliminary questionnaire results"""

    new_df = pd.DataFrame({
                        'ID':               pd.Series(dtype='str'),
                        'Communication':    pd.Series(dtype='str'),
                        'Order':            pd.Series(dtype='str'), 
                        'Age':              pd.Series(dtype='str'),
                        'Gender':           pd.Series(dtype='str'),
                        'Gaming Frequency': pd.Series(dtype='str'),
                    })

    for _, row in df.iterrows():

        # Find age group
        age = ''
        if row['Under20'] == 1:
            age = 'Under 20'
        elif row['20-29'] == 1:
            age = '20-29'
        elif row['30-39'] == 1:
            age = '30-39'
        elif row['40-49'] == 1:
            age = '40-49'
        elif row['50-59'] == 1:
            age = '50-59'
        elif row['60&Over'] == 1:
            age = '60 & Over'
        elif row['PNTS'] == 1:
            age = 'Prefer Not to Say'

        # Find gaming frequency
        gaming_frequency = ''
        if row['Daily'] == 1:
            gaming_frequency = 'Daily'
        elif row['Weekly'] == 1:
            gaming_frequency = 'Weekly'
        elif row['Monthly'] == 1:
            gaming_frequency = 'Monthly'
        elif row['Past'] == 1:
            gaming_frequency = 'Past'
        elif row['Never'] == 1:
            gaming_frequency = 'Never'

        d = pd.DataFrame({'ID': [row['ID']], 'Communication': [row['Communication']], 'Order': [row['Order']], 'Age': [age], 'Gender': [row['Gender']], 'Gaming Frequency': [gaming_frequency]})
        new_df = pd.concat([new_df, d], ignore_index=True, axis=0)

    print(new_df)

    # Plot
    fig, axes = plt.subplots(nrows=2, ncols=1)

    age_group_size = new_df.groupby('Age').size()
    gaming_frequency_size = new_df.groupby('Gaming Frequency').size()

    age_groups_count = {
                            'Under 20': age_group_size['Under 20'], 
                            '20-29': age_group_size['20-29'],
                            '30-39': age_group_size['30-39'],
                            '40-49': age_group_size['40-49'],
                            '50-59': age_group_size['50-59'],
                            '60 & Over': age_group_size['60 & Over'],
                            # 'Prefer Not to Say': age_group_size['Prefer Not to Say'],
                        }

    gaming_frequency_count = {
                                'Daily': gaming_frequency_size['Daily'],
                                'Weekly': gaming_frequency_size['Weekly'],
                                'Monthly': gaming_frequency_size['Monthly'],
                                'Past': gaming_frequency_size['Past'],
                                'Never': gaming_frequency_size['Never'],
                            }

    # Reference: https://towardsdatascience.com/stacked-bar-charts-with-pythons-matplotlib-f4020e4eb4a7
    # TODO: find better colormap
    age_colors = ['#ffff33', '#ff7f00', '#984ea3', '#4daf4a', '#377eb8', '#e41a1c']
    gaming_colors = ['#253494', '#2c7fb8', '#41b6c4', '#7fcdbb', '#c7e9b4']

    age_labels = ['Under 20', '20-29', '30-39', '40-49', '50-59', '60 & Over']
    gaming_labels = ['Daily', 'Weekly', 'Monthly', 'Past', 'Never']

    left = [0]
    for i, (key, value) in enumerate(age_groups_count.items()):
        axes[0].barh(['Age'], value, left = left, color=age_colors[i])
        left = left + value
        print(left)

    left = [0]
    for i, (key, value) in enumerate(gaming_frequency_count.items()):
        axes[1].barh(['Gaming\nFrequency'], value, left = left, color=gaming_colors[i])
        left = left + value
        print(left)

    # title, legend, labels
    # axes[0].set_title('Participant age groups\n', loc='center')
    axes[0].legend(age_labels, bbox_to_anchor=([1.05, 1, 0, 0]), ncol=2, frameon=True)

    # axes[1].set_title('Gaming frequency\n', loc='center')
    axes[1].legend(gaming_labels, bbox_to_anchor=([1.05, 1, 0, 0]), ncol=2, frameon=True)
    axes[1].set_xlabel('Participants')

    # remove spines
    for i in range(2):
        axes[i].spines['right'].set_visible(False)
        # axes[i].spines['left'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['bottom'].set_visible(False)

        # adjust limits and draw grid lines
        # plt.ylim(-0.5, axes[i].get_yticks()[-1] + 0.5)
        axes[i].set_axisbelow(True)
        axes[i].xaxis.grid(color='gray', linestyle='dashed')

    fig.tight_layout()

    plt.show()


def plot_gaming_frequency_vs_points(stats, df, include_partial_points=False):
    """Plot the user's gaming frequency against points scored"""

    # Need to match gaming frequency and points using id

    # Store score and two user id
    # Temp store score and user in a dict (key: user, value: [trial1_score, trial2_score])
    user_score = {}

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:
                print(trial.users, trial.totalPoints)

                # Add entry for each user to dict
                for user in trial.users:
                    if user not in user_score:
                        user_score[user] = {}

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)
                points_scored = trial.totalPoints

                if include_partial_points:

                    partial_points_scored = 0

                    # Check final timestep for any partially completed tasks
                    for task in trial.data[trial.totalTime]['log'].tasks:
                        factor = (300. - 50) / (12. - 1)
                        init_demand = math.floor((task.requiredRobots * factor) + (50 - factor))

                        if task.demand < init_demand:
                            # Calculate points for each unfinished task
                            partial_points_scored += (float(task.demand) / init_demand) * task.requiredRobots
                            # print('Partially completed task of size', task.requiredRobots, '(', task.demand, '/', init_demand, ') Adding', (float(task.demand) / init_demand) * task.requiredRobots)

                    print('Total partial points', partial_points_scored)

                    # Add partially completed tasks to the score
                    points_scored += partial_points_scored

                trial_num = 'trial'

                if int(cond[-1]) == 1:
                    if send == 'no_send':
                        trial_num += '1'
                    else:
                        trial_num += '2'
                else:
                    if send == 'no_send':
                        trial_num += '2'
                    else:
                        trial_num += '1'

                for user in trial.users:
                    user_score[user][trial_num] = points_scored

    pprint.pprint(user_score)

    # Store gaming frequency per user

    # Split data by experiment order
    order1_df = df[df['Order'] == 1]
    order2_df = df[df['Order'] == 2]

    # Split each data row so that there is one row for each trial
    order1_no_send_df = order1_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order1_no_send_df['Condition'] = 'NRS'

    order1_send_df = order1_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_1 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order1_send_df)]
    order1_send_df.columns = column_labels_1
    order1_send_df['Condition'] = 'RS'

    order1_df = pd.concat([order1_no_send_df, order1_send_df])
    # print(order1_df)

    order2_no_send_df = order2_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_2 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order2_no_send_df)]
    order2_no_send_df.columns = column_labels_2
    order2_no_send_df['Condition'] = 'NRS'

    order2_send_df = order2_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order2_send_df['Condition'] = 'RS'

    order2_df = pd.concat([order2_no_send_df, order2_send_df])
    # print(order2_df)

    df = pd.concat([order1_df, order2_df])
    print(df)

    # Calculate the average workload score for each trial
    average_task_load = []
    for index, row in df.iterrows():
        total = row['Q1'] + row['Q2'] + row['Q3'] + row['Q4'] + row['Q5'] + row['Q6']
        average = total / 6.0
        average_task_load.append(average)

    df.insert(len(df.columns), 'Average Taskload', average_task_load)

    gaming_frequency = []
    for index, row in df.iterrows():
        if row['Daily'] == 1:
            gaming_frequency.append('Daily')
        elif row['Weekly'] == 1:
            gaming_frequency.append('Weekly')
        elif row['Monthly'] == 1:
            gaming_frequency.append('Monthly')
        elif row['Past'] == 1:
            gaming_frequency.append('Past')
        elif row['Never'] == 1:
            gaming_frequency.append('Never')

    df.insert(len(df.columns), 'Gaming Frequency', gaming_frequency)

    # Find the score of each trial
    trial_score = []
    for index, row in df.iterrows():
        user = row['ID']
        trial_num = 'trial' + str(row['Order'])
        trial_score.append(user_score[user][trial_num])

    df.insert(len(df.columns), 'Points', trial_score)

    print(df)
    df.to_csv(join(RESULTS_DIR, 'gaming_frequency.csv'), index=False)

    # Plot
    fig, axes = plt.subplots(nrows=1, ncols=1)

    sns.boxplot(data=df, ax=axes, x='Gaming Frequency', y='Average Taskload', order=['Daily', 'Weekly', 'Monthly', 'Past', 'Never'], fliersize=0)

    sns.stripplot(data=df, ax=axes, x='Gaming Frequency', y='Average Taskload', order=['Daily', 'Weekly', 'Monthly', 'Past', 'Never'], size=6, linewidth=2, dodge=True)

    fig.tight_layout()

    # Plot
    fig2, axes2 = plt.subplots(nrows=1, ncols=1)

    sns.boxplot(data=df, ax=axes2, x='Gaming Frequency', y='Points', order=['Daily', 'Weekly', 'Monthly', 'Past', 'Never'], fliersize=0)

    sns.stripplot(data=df, ax=axes2, x='Gaming Frequency', y='Points', order=['Daily', 'Weekly', 'Monthly', 'Past', 'Never'], size=6, linewidth=2, dodge=True)

    # # plt.setp(axes, xlim=[0,50])

    fig2.tight_layout()

    plt.show()

def plot_task_load(df):
    """Plot the user's experienced task load"""
    
    # Split data by experiment order
    order1_df = df[df['Order'] == 1]
    order2_df = df[df['Order'] == 2]

    # Split each data row so that there is one row for each trial
    order1_no_send_df = order1_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order1_no_send_df['Condition'] = 'NRS'

    order1_send_df = order1_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_1 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order1_send_df)]
    order1_send_df.columns = column_labels_1
    order1_send_df['Condition'] = 'RS'

    order1_df = pd.concat([order1_no_send_df, order1_send_df])
    # print(order1_df)

    order2_no_send_df = order2_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_2 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order2_no_send_df)]
    order2_no_send_df.columns = column_labels_2
    order2_no_send_df['Condition'] = 'NRS'

    order2_send_df = order2_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order2_send_df['Condition'] = 'RS'

    order2_df = pd.concat([order2_no_send_df, order2_send_df])
    # print(order2_df)

    df = pd.concat([order1_df, order2_df])
    print(df)

    # Plot average task load index 
    fig, axes = plt.subplots(nrows=2, ncols=3)

    sns.violinplot(data=df, ax=axes[0,0], x='Communication', y='Q1', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], palette='Set2')
    sns.violinplot(data=df, ax=axes[0,1], x='Communication', y='Q2', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], palette='Set2')
    sns.violinplot(data=df, ax=axes[0,2], x='Communication', y='Q3', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], palette='Set2')
    sns.violinplot(data=df, ax=axes[1,0], x='Communication', y='Q4', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], palette='Set2')
    sns.violinplot(data=df, ax=axes[1,1], x='Communication', y='Q5', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], palette='Set2')
    sns.violinplot(data=df, ax=axes[1,2], x='Communication', y='Q6', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], palette='Set2')

    sns.stripplot(data=df, ax=axes[0,0], x='Communication', y='Q1', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[0,1], x='Communication', y='Q2', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[0,2], x='Communication', y='Q3', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[1,0], x='Communication', y='Q4', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[1,1], x='Communication', y='Q5', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[1,2], x='Communication', y='Q6', order=['DIR', 'IND'], hue='Condition', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)

    # Set plot titles
    axes[0,0].set_title('Q1. Mental Demand')
    axes[0,1].set_title('Q2. Physical Demand')
    axes[0,2].set_title('Q3. Temporal Demand')
    axes[1,0].set_title('Q4. Performance')
    axes[1,1].set_title('Q5. Effort')
    axes[1,2].set_title('Q6. Frustration')

    # Set y-axis labels
    for i, ax in enumerate(axes.flat):
        ax.set_ylabel("Participant's Rating")

    plt.setp(axes, ylim=[0,7.9])

    fig.tight_layout()

    # plt.show()

    fig2, axes2 = plt.subplots(nrows=2, ncols=3)

    sns.violinplot(data=df, ax=axes2[0,0], x='Condition', y='Q1', order=['RS', 'NRS'], palette='Set1')
    sns.violinplot(data=df, ax=axes2[0,1], x='Condition', y='Q2', order=['RS', 'NRS'], palette='Set1')
    sns.violinplot(data=df, ax=axes2[0,2], x='Condition', y='Q3', order=['RS', 'NRS'], palette='Set1')
    sns.violinplot(data=df, ax=axes2[1,0], x='Condition', y='Q4', order=['RS', 'NRS'], palette='Set1')
    sns.violinplot(data=df, ax=axes2[1,1], x='Condition', y='Q5', order=['RS', 'NRS'], palette='Set1')
    sns.violinplot(data=df, ax=axes2[1,2], x='Condition', y='Q6', order=['RS', 'NRS'], palette='Set1')

    sns.stripplot(data=df, ax=axes2[0,0], x='Condition', y='Q1', order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes2[0,1], x='Condition', y='Q2', order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes2[0,2], x='Condition', y='Q3', order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes2[1,0], x='Condition', y='Q4', order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes2[1,1], x='Condition', y='Q5', order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes2[1,2], x='Condition', y='Q6', order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)

    # Set plot titles
    axes2[0,0].set_title('Q1. Mental Demand')
    axes2[0,1].set_title('Q2. Physical Demand')
    axes2[0,2].set_title('Q3. Temporal Demand')
    axes2[1,0].set_title('Q4. Performance')
    axes2[1,1].set_title('Q5. Effort')
    axes2[1,2].set_title('Q6. Frustration')

    # Set y-axis labels
    questions = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']
    for i, ax in enumerate(axes2.flat):
        ax.set_ylabel("Participant's Rating")

        # statistical annotation
        pairs = [('RS', 'NRS')]
        annotator = Annotator(ax, pairs, data=df, x='Condition', y=questions[i], order=['RS', 'NRS'])
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
        annotator.apply_and_annotate()

    plt.setp(axes2, ylim=[0,7.9])

    fig2.tight_layout()

    plt.show()


def plot_global_task_load(df):
    """Plot the user's global experienced task load"""
    
    # Split data by experiment order
    order1_df = df[df['Order'] == 1]
    order2_df = df[df['Order'] == 2]

    # Split each data row so that there is one row for each trial
    order1_no_send_df = order1_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order1_no_send_df['Condition'] = 'NRS'

    order1_send_df = order1_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_1 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order1_send_df)]
    order1_send_df.columns = column_labels_1
    order1_send_df['Condition'] = 'RS'

    order1_df = pd.concat([order1_no_send_df, order1_send_df])
    # print(order1_df)

    order2_no_send_df = order2_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_2 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order2_no_send_df)]
    order2_no_send_df.columns = column_labels_2
    order2_no_send_df['Condition'] = 'NRS'

    order2_send_df = order2_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order2_send_df['Condition'] = 'RS'

    order2_df = pd.concat([order2_no_send_df, order2_send_df])
    # print(order2_df)

    df = pd.concat([order1_df, order2_df])
    print(df)

    # Calculate the average workload score for each trial
    average_task_load = []
    for index, row in df.iterrows():
        total = row['Q1'] + row['Q2'] + row['Q3'] + row['Q4'] + row['Q5'] + row['Q6']
        average = total / 6.0
        average_task_load.append(average)

    df.insert(len(df.columns), 'Average Taskload', average_task_load)

    pprint.pprint(df)
    df.to_csv(join(RESULTS_DIR, 'global_taskload.csv'), index=False)

    # Plot average task load index 
    fig, axes = plt.subplots(nrows=1, ncols=3)

    sns.boxplot(data=df, ax=axes[0], x='Communication', y='Average Taskload', order=['DIR', 'IND'], palette='Set2')
    sns.boxplot(data=df, ax=axes[1], x='Condition', y='Average Taskload', order=['RS', 'NRS'], palette='Set1')
    sns.boxplot(data=df, ax=axes[2], x='Communication', y='Average Taskload', order=['DIR', 'IND'], hue='Condition', hue_order=['RS', 'NRS'], palette='Set1')

    sns.stripplot(data=df, ax=axes[0], x='Communication', y='Average Taskload', order=['DIR', 'IND'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[1], x='Condition', y='Average Taskload', order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    g = sns.stripplot(data=df, ax=axes[2], x='Communication', y='Average Taskload', order=['DIR', 'IND'], hue='Condition', hue_order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # Set y-axis labels
    # for i, ax in enumerate(axes.flat):
    #     ax.set_ylabel("Participant's Rating")

    plt.setp(axes, ylim=[0.5,7.5])

    fig.tight_layout()

    # Plot average task load index 
    fig2, axes2 = plt.subplots(nrows=1, ncols=1)

    # sns.boxplot(data=df, ax=axes[0], x='Communication', y='Average Taskload', order=['DIR', 'IND'], palette='Set2')
    sns.boxplot(data=df, ax=axes2, x='Condition', y='Average Taskload', order=['RS', 'NRS'], palette='Set1')
    # sns.boxplot(data=df, ax=axes[2], x='Communication', y='Average Taskload', order=['DIR', 'IND'], hue='Condition', hue_order=['RS', 'NRS'], palette='Set1')

    # sns.stripplot(data=df, ax=axes[0], x='Communication', y='Average Taskload', order=['DIR', 'IND'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes2, x='Condition', y='Average Taskload', order=['RS', 'NRS'], size=6, color='.25', linewidth=0, dodge=True, jitter=True)
    # g = sns.stripplot(data=df, ax=axes[2], x='Communication', y='Average Taskload', order=['DIR', 'IND'], hue='Condition', hue_order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    # g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    x_coords = np.array([df.query('Condition=="RS"').Condition, df.query('Condition=="NRS"').Condition])
    y_coords = np.array([df.query('Condition=="RS"')['Average Taskload'], df.query('Condition=="NRS"')['Average Taskload']])
    # print(x_coords)
    # print(y_coords)

    mask_y_coords = np.isnan(y_coords)
    mask_y_coords = np.logical_or(mask_y_coords[0], mask_y_coords[1])
    mask_y_coords = np.logical_not(mask_y_coords)
    # print(mask_y_coords)

    # y_coords[0] = y_coords[0][mask_y_coords]
    # y_coords[1] = y_coords[1][mask_y_coords]
    y_coords = np.array([y_coords[0][mask_y_coords], y_coords[1][mask_y_coords]])
    x_coords = np.array([x_coords[0][mask_y_coords], x_coords[1][mask_y_coords]])
    # print(y_coords)
    # print(x_coords)

    # Loop x-coords annd y-coords to create a list of pairs
    # example: lines = [[(0, 11.44234246), (1, 12.05103481)]]
    # lines = []
    # for i in range(len(x_coords[0])):
    #     pair = [(0, y_coords[0][i]), (1, y_coords[1][i])]
    #     lines.append(pair)

    # lc = LineCollection(lines)
    # lc.set_color((0.5, 0.5, 0.5))
    # axes2.add_collection(lc)

    # Remove id with at least one nan
    # Loop dataframe and keep track of id with nan
    # Delete rows with id from df
    id_to_remove = []
    for _, row in df.iterrows():
        # print(row['ID'], row['Average Taskload'])
        if pd.isna(row['Average Taskload']):
            print('TRUE', row.ID)
            id_to_remove.append(row['ID'])

    print(id_to_remove)
    df_removed_invalid = df[~df.ID.isin(id_to_remove)]
    print(df_removed_invalid)

    # statistical annotation
    pairs = [('RS', 'NRS')]
    annotator = Annotator(axes2, pairs, data=df_removed_invalid, x='Condition', y='Average Taskload', order=['RS','NRS'])
    annotator.configure(test='Wilcoxon', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    plt.setp(axes2, ylim=[0.5,7.5])

    fig2.tight_layout()

    plt.show()


def plot_global_task_load_vs_points(stats, df, include_partial_points=False):
    """Plot the user's global experienced task load against points scored"""

    # Need to match task load and points using id

    # Store score and two user id
    # Temp store score and user in a dict (key: user, value: [trial1_score, trial2_score])
    user_score = {}

    for cond, cond_stats in stats.items():
        for send, send_stats in cond_stats.items():
            print('---', cond, send, '---')

            for trial in send_stats:
                print(trial.users, trial.totalPoints)

                # Add entry for each user to dict
                for user in trial.users:
                    if user not in user_score:
                        user_score[user] = {}

                # Store result to dataframe
                cond_value, send_value = get_config(cond, send)
                points_scored = trial.totalPoints

                if include_partial_points:

                    partial_points_scored = 0

                    # Check final timestep for any partially completed tasks
                    for task in trial.data[trial.totalTime]['log'].tasks:
                        factor = (300. - 50) / (12. - 1)
                        init_demand = math.floor((task.requiredRobots * factor) + (50 - factor))

                        if task.demand < init_demand:
                            # Calculate points for each unfinished task
                            partial_points_scored += (float(task.demand) / init_demand) * task.requiredRobots
                            # print('Partially completed task of size', task.requiredRobots, '(', task.demand, '/', init_demand, ') Adding', (float(task.demand) / init_demand) * task.requiredRobots)

                    print('Total partial points', partial_points_scored)

                    # Add partially completed tasks to the score
                    points_scored += partial_points_scored

                trial_num = 'trial'

                if int(cond[-1]) == 1:
                    if send == 'no_send':
                        trial_num += '1'
                    else:
                        trial_num += '2'
                else:
                    if send == 'no_send':
                        trial_num += '2'
                    else:
                        trial_num += '1'

                for user in trial.users:
                    user_score[user][trial_num] = points_scored

    pprint.pprint(user_score)

    # Store global task load per user

    # Split data by experiment order
    order1_df = df[df['Order'] == 1]
    order2_df = df[df['Order'] == 2]

    # Split each data row so that there is one row for each trial
    order1_no_send_df = order1_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order1_no_send_df['Condition'] = 'NRS'

    order1_send_df = order1_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_1 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order1_send_df)]
    order1_send_df.columns = column_labels_1
    order1_send_df['Condition'] = 'RS'

    order1_df = pd.concat([order1_no_send_df, order1_send_df])
    # print(order1_df)

    order2_no_send_df = order2_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_2 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order2_no_send_df)]
    order2_no_send_df.columns = column_labels_2
    order2_no_send_df['Condition'] = 'NRS'

    order2_send_df = order2_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order2_send_df['Condition'] = 'RS'

    order2_df = pd.concat([order2_no_send_df, order2_send_df])
    # print(order2_df)

    df = pd.concat([order1_df, order2_df])
    print(df)

    # Calculate the average workload score for each trial
    average_task_load = []
    for index, row in df.iterrows():
        total = row['Q1'] + row['Q2'] + row['Q3'] + row['Q4'] + row['Q5'] + row['Q6']
        average = total / 6.0
        average_task_load.append(average)

    df.insert(len(df.columns), 'Average Taskload', average_task_load)

    # Find the score of each trial
    trial_score = []
    for index, row in df.iterrows():
        user = row['ID']
        trial_num = 'trial' + str(row['Order'])
        trial_score.append(user_score[user][trial_num])

    df.insert(len(df.columns), 'Points', trial_score)

    print(df)
    df.to_csv(join(RESULTS_DIR, 'global_taskload.csv'), index=False)

    # Plot points scored against the number of robots sent between leaders
    fig, axes = plt.subplots(nrows=1, ncols=1)

    sns.scatterplot(data=df, ax=axes, x='Average Taskload', y='Points', hue='Communication', hue_order=['DIR', 'IND'], style='Communication')

    # plt.setp(axes, xlim=[0,50])

    plt.show()


def plot_situational_awareness(df):
    """Plot the user's situational awareness"""

    # Split data by experiment order
    order1_df = df[df['Order'] == 1]
    order2_df = df[df['Order'] == 2]

    # Split each data row so that there is one row for each trial
    order1_no_send_df = order1_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order1_no_send_df['Send'] = 'No'

    order1_send_df = order1_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_1 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order1_send_df)]
    order1_send_df.columns = column_labels_1
    order1_send_df['Send'] = 'Yes'

    order1_df = pd.concat([order1_no_send_df, order1_send_df])
    # print(order1_df)

    order2_no_send_df = order2_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_2 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order2_no_send_df)]
    order2_no_send_df.columns = column_labels_2
    order2_no_send_df['Send'] = 'NRS'

    order2_send_df = order2_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order2_send_df['Send'] = 'RS'

    order2_df = pd.concat([order2_no_send_df, order2_send_df])
    # print(order2_df)

    df = pd.concat([order1_df, order2_df])
    print(df)

    # Plot average task load index 
    fig, axes = plt.subplots(nrows=3, ncols=3)

    sns.boxplot(data=df, ax=axes[0,0], x='Communication', y='Q7', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], palette='Set2')
    sns.boxplot(data=df, ax=axes[0,1], x='Communication', y='Q8', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], palette='Set2')
    sns.boxplot(data=df, ax=axes[0,2], x='Communication', y='Q9', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], palette='Set2')
    sns.boxplot(data=df, ax=axes[1,0], x='Communication', y='Q10', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], palette='Set2')
    sns.boxplot(data=df, ax=axes[1,1], x='Communication', y='Q11', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], palette='Set2')
    sns.boxplot(data=df, ax=axes[1,2], x='Communication', y='Q12', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], palette='Set2')
    sns.boxplot(data=df, ax=axes[2,0], x='Communication', y='Q13', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], palette='Set2')
    sns.boxplot(data=df, ax=axes[2,1], x='Communication', y='Q14', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], palette='Set2')
    sns.boxplot(data=df, ax=axes[2,2], x='Communication', y='Q15', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], palette='Set2')

    sns.stripplot(data=df, ax=axes[0,0], x='Communication', y='Q7', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[0,1], x='Communication', y='Q8', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[0,2], x='Communication', y='Q9', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[1,0], x='Communication', y='Q10', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[1,1], x='Communication', y='Q11', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[1,2], x='Communication', y='Q12', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[2,0], x='Communication', y='Q13', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[2,1], x='Communication', y='Q14', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[2,2], x='Communication', y='Q15', order=['DIR', 'IND'], hue='Send', hue_order=['RS','NRS'], size=6, palette='Set2', linewidth=2, dodge=True)

    # Set plot titles
    axes[0,0].set_title('Q7. Instabiliity of Situation')
    axes[0,1].set_title('Q8. Complexity of Situation')
    axes[0,2].set_title('Q9. Variability of Situation')
    axes[1,0].set_title('Q10. Arousal')
    axes[1,1].set_title('Q11. Concentration of Attention')
    axes[1,2].set_title('Q12. Division of Attention')
    axes[2,0].set_title('Q13. Spare Mental Capacity')
    axes[2,1].set_title('Q14. Information Quantity')
    axes[2,2].set_title('Q15. Familiarity with Situation')

    # Set y-axis labels
    for i, ax in enumerate(axes.flat):
        ax.set_ylabel("Participant's Rating")

    plt.setp(axes, ylim=[0.5,7.5])

    fig.tight_layout()

    plt.show()


def plot_global_situational_awareness(df):
    """Plot the user's global situational awareness"""

    # Split data by experiment order
    order1_df = df[df['Order'] == 1]
    order2_df = df[df['Order'] == 2]

    # Split each data row so that there is one row for each trial
    order1_no_send_df = order1_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order1_no_send_df['Condition'] = 'NRS'

    order1_send_df = order1_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_1 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order1_send_df)]
    order1_send_df.columns = column_labels_1
    order1_send_df['Condition'] = 'RS'

    order1_df = pd.concat([order1_no_send_df, order1_send_df])
    # print(order1_df)

    order2_no_send_df = order2_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_2 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order2_no_send_df)]
    order2_no_send_df.columns = column_labels_2
    order2_no_send_df['Condition'] = 'NRS'

    order2_send_df = order2_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order2_send_df['Condition'] = 'RS'

    order2_df = pd.concat([order2_no_send_df, order2_send_df])
    # print(order2_df)

    df = pd.concat([order1_df, order2_df])
    print(df)

    # Calculate the situational awareness score for each trial
    average_sa = []
    for index, row in df.iterrows():
        total = row['Q7'] + row['Q8'] + row['Q9'] + row['Q10'] + row['Q11'] + row['Q12'] + row['Q13'] + row['Q14'] + row['Q15']
        average = total / 9.0
        average_sa.append(average)

    df.insert(len(df.columns), 'Average Situational Awareness', average_sa)

    print(df)
    df.to_csv(join(RESULTS_DIR, 'global_situational_awareness.csv'), index=False)

    # Plot average situational awareness
    fig, axes = plt.subplots(nrows=1, ncols=3)

    sns.boxplot(data=df, ax=axes[0], x='Communication', y='Average Situational Awareness', order=['DIR', 'IND'], palette='Set2')
    sns.boxplot(data=df, ax=axes[1], x='Condition', y='Average Situational Awareness', order=['RS', 'NRS'], palette='Set1')
    sns.boxplot(data=df, ax=axes[2], x='Communication', y='Average Situational Awareness', order=['DIR', 'IND'], hue='Condition', hue_order=['RS', 'NRS'], palette='Set1')

    sns.stripplot(data=df, ax=axes[0], x='Communication', y='Average Situational Awareness', order=['DIR', 'IND'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[1], x='Condition', y='Average Situational Awareness', order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    g = sns.stripplot(data=df, ax=axes[2], x='Communication', y='Average Situational Awareness', order=['DIR', 'IND'], hue='Condition', hue_order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # Set y-axis labels
    for i, ax in enumerate(axes.flat):
        ax.set_ylabel("Participant's Rating")

    plt.setp(axes, ylim=[0.5,7.5])

    fig.tight_layout()

    plt.show()


def plot_teammate_understanding(df):
    """Plot the user's understading of their teammate's behavior"""

    # Split data by experiment order
    order1_df = df[df['Order'] == 1]
    order2_df = df[df['Order'] == 2]

    # Split each data row so that there is one row for each trial
    order1_no_send_df = order1_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order1_no_send_df['Condition'] = 'NRS'

    order1_send_df = order1_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_1 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order1_send_df)]
    order1_send_df.columns = column_labels_1
    order1_send_df['Condition'] = 'RS'

    order1_df = pd.concat([order1_no_send_df, order1_send_df])
    # print(order1_df)

    order2_no_send_df = order2_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_2 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order2_no_send_df)]
    order2_no_send_df.columns = column_labels_2
    order2_no_send_df['Condition'] = 'NRS'

    order2_send_df = order2_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order2_send_df['Condition'] = 'RS'

    order2_df = pd.concat([order2_no_send_df, order2_send_df])
    # print(order2_df)

    df = pd.concat([order1_df, order2_df])
    print(df)

    # Plot understanding of teammate
    fig, axes = plt.subplots(nrows=1, ncols=3)

    sns.boxplot(data=df, ax=axes[0], x='Communication', y='Q16', order=['DIR', 'IND'], palette='Set2')
    sns.boxplot(data=df, ax=axes[1], x='Condition', y='Q16', order=['RS', 'NRS'], palette='Set1')
    sns.boxplot(data=df, ax=axes[2], x='Communication', y='Q16', order=['DIR', 'IND'], hue='Condition', hue_order=['RS', 'NRS'], palette='Set1')

    sns.stripplot(data=df, ax=axes[0], x='Communication', y='Q16', order=['DIR', 'IND'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[1], x='Condition', y='Q16', order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    g = sns.stripplot(data=df, ax=axes[2], x='Communication', y='Q16', order=['DIR', 'IND'], hue='Condition', hue_order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # Set y-axis labels
    for i, ax in enumerate(axes.flat):
        ax.set_ylabel("Understanding of teammate's actions")

    plt.setp(axes, ylim=[0.5,7.5])

    fig.tight_layout()

    plt.show()


def plot_robot_understanding(df):
    """Plot the user's understading of the robots' behavior"""

    # Split data by experiment order
    order1_df = df[df['Order'] == 1]
    order2_df = df[df['Order'] == 2]

    # Split each data row so that there is one row for each trial
    order1_no_send_df = order1_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order1_no_send_df['Condition'] = 'NRS'

    order1_send_df = order1_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_1 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order1_send_df)]
    order1_send_df.columns = column_labels_1
    order1_send_df['Condition'] = 'RS'

    order1_df = pd.concat([order1_no_send_df, order1_send_df])
    # print(order1_df)

    order2_no_send_df = order2_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_2 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order2_no_send_df)]
    order2_no_send_df.columns = column_labels_2
    order2_no_send_df['Condition'] = 'NRS'

    order2_send_df = order2_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order2_send_df['Condition'] = 'RS'

    order2_df = pd.concat([order2_no_send_df, order2_send_df])
    # print(order2_df)

    df = pd.concat([order1_df, order2_df])
    print(df)

    # Plot understanding of robots
    fig, axes = plt.subplots(nrows=1, ncols=3)

    sns.boxplot(data=df, ax=axes[0], x='Communication', y='Q17', order=['DIR', 'IND'], palette='Set2')
    sns.boxplot(data=df, ax=axes[1], x='Condition', y='Q17', order=['RS', 'NRS'], palette='Set1')
    sns.boxplot(data=df, ax=axes[2], x='Communication', y='Q7', order=['DIR', 'IND'], hue='Condition', hue_order=['RS', 'NRS'], palette='Set1')

    sns.stripplot(data=df, ax=axes[0], x='Communication', y='Q17', order=['DIR', 'IND'], size=6, palette='Set2', linewidth=2, dodge=True)
    sns.stripplot(data=df, ax=axes[1], x='Condition', y='Q17', order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    g = sns.stripplot(data=df, ax=axes[2], x='Communication', y='Q17', order=['DIR', 'IND'], hue='Condition', hue_order=['RS', 'NRS'], size=6, palette='Set1', linewidth=2, dodge=True)
    g.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)

    # Set y-axis labels
    for i, ax in enumerate(axes.flat):
        ax.set_ylabel("Understanding of robot's actions")

    plt.setp(axes, ylim=[0.5,7.5])

    fig.tight_layout()

    plt.show()


def plot_questionnaire_responses(df):
    """Plot the user's response to other questions"""

    # Split data by experiment order
    order1_df = df[df['Order'] == 1]
    order2_df = df[df['Order'] == 2]

    # Split each data row so that there is one row for each trial
    order1_no_send_df = order1_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order1_no_send_df['Condition'] = 'NRS'

    order1_send_df = order1_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_1 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order1_send_df)]
    order1_send_df.columns = column_labels_1
    order1_send_df['Condition'] = 'RS'

    order1_df = pd.concat([order1_no_send_df, order1_send_df])
    # print(order1_df)

    order2_no_send_df = order2_df.drop(['Q' + str(i) for i in range(1,18)], axis=1)
    column_labels_2 = [label.replace('.1','') if label[0] == 'Q' else label for label in list(order2_no_send_df)]
    order2_no_send_df.columns = column_labels_2
    order2_no_send_df['Condition'] = 'NRS'

    order2_send_df = order2_df.drop(['Q' + str(i) + '.1' for i in range(1,18)], axis=1)
    order2_send_df['Condition'] = 'RS'

    order2_df = pd.concat([order2_no_send_df, order2_send_df])
    # print(order2_df)

    new_df = pd.concat([order1_df, order2_df])
    print(df)

    # Did you understand the teammate's behavior?
    q16_comm_size = new_df.query('Communication=="DIR"').groupby('Q16').size()
    q17_comm_size = new_df.query('Communication=="IND"').groupby('Q16').size()
    # Did you understand the robot's behavior?
    q16_cond_size = new_df.query('Condition=="RS"').groupby('Q17').size()
    q17_cond_size = new_df.query('Condition=="NRS"').groupby('Q17').size()
    # Interface clear to understand?
    q18_size = df.groupby('Q18').size()
    # Was sharing robots useful?
    q19_size = df.groupby('Q19').size()

    q16_comm_dict = q16_comm_size.to_dict(defaultdict(int))
    q17_comm_dict = q17_comm_size.to_dict(defaultdict(int))
    q16_cond_dict = q16_cond_size.to_dict(defaultdict(int))
    q17_cond_dict = q17_cond_size.to_dict(defaultdict(int))
    q18_dict = q18_size.to_dict(defaultdict(int))
    q19_dict = q19_size.to_dict(defaultdict(int))

    q16_comm_data = []
    q17_comm_data = []
    q16_cond_data = []
    q17_cond_data = []
    q18_data = []
    q19_data = []

    print(q16_comm_dict)
    print(q17_comm_dict)
    print(q16_cond_dict)
    print(q17_cond_dict)
    print(q18_dict)
    print(q19_dict)

    for i in range(1,8):
        q16_comm_data.append(q16_comm_dict[i])
        q17_comm_data.append(q17_comm_dict[i])
        q16_cond_data.append(q16_cond_dict[i])
        q17_cond_data.append(q17_cond_dict[i])
        q18_data.append(q18_dict[i])
        q19_data.append(q19_dict[i])

    print(q16_comm_data)
    print(q17_comm_data)
    print(q16_cond_data)
    print(q17_cond_data)
    print(q18_data)
    print(q19_data)

    # Reference: https://stackoverflow.com/a/69976552
    category_names = ['1 (Strongly disagree)', '2', '3', '4 (Neutral)', '5', '6', '7 (Strongly agree)']
    results_labels = [
        'Did you understand what your\n' + r'$\bf{teammate}$ was doing? $\it{(DIR, N=56)}$',
        'Did you understand what your\n' + r'$\bf{teammate}$ was doing? $\it{(IND, N=48)}$',
        # 'Did you understand what\n' + r'your $\bf{teammate}$ was doing? (RS)',
        # 'Did you understand what\n' + r'your $\bf{teammate}$ was doing? (NRS)',
        # 'Did you understand what\n' + r'the $\bf{robots}$ were doing? (DIR)',
        # 'Did you understand what\n' + r'the $\bf{robots}$ were doing? (IND)',
        'Did you understand what the\n' + r'$\bf{robots}$ were doing? $\it{(RS, N=52)}$',
        'Did you understand what the\n' + r'$\bf{robots}$ were doing? $\it{(NRS, N=52)}$',

        'Was the interface clear to\nunderstand? ' + r'$\it{(N=52)}$',
        'Did you find the ability to share\nrobots useful? ' + r'$\it{(N=52)}$'
    ]
    results = {
        results_labels[0]: q16_comm_data,
        results_labels[1]: q17_comm_data,
        results_labels[2]: q16_cond_data,
        results_labels[3]: q17_cond_data,
        results_labels[4]: q18_data,
        results_labels[5]: q19_data,
    }

    def survey(results, category_names):
        """
        Parameters
        ----------
        results : dict
            A mapping from question labels to a list of answers per category.
            It is assumed all lists contain the same number of entries and that
            it matches the length of *category_names*. The order is assumed
            to be from 'Strongly disagree' to 'Strongly agree'
        category_names : list of str
            The category labels.
        """
        
        labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        middle_index = data.shape[1]//2
        offsets = data[:, range(middle_index)].sum(axis=1) + data[:, middle_index]/2
        
        # Color Mapping
        category_colors = plt.get_cmap('RdBu')(
            np.linspace(0.2, 0.8, data.shape[1]))
        print(category_colors)
        # category_colors[int((len(category_colors) - 1)/2),:] = np.array([0.95,0.95,0.95,1])
        # print(category_colors)

        fig, ax = plt.subplots(figsize=(10, 2.4))
        
        # Font
        font = FontProperties()
        # font.set_family('serif')
        # font.set_name('Times New Roman')
        font.set_size(7)

        font2 = FontProperties()
        # font2.set_family('serif')
        # font2.set_name('Times New Roman')
        font2.set_size(8)

        for label in ax.get_xticklabels():
            label.set_fontproperties(font2)
        for label in ax.get_yticklabels():
            label.set_fontproperties(font)

        # Add Zero Reference Line
        ax.axvline(0, linewidth=0.5, linestyle='-', color='black', alpha=0.75, zorder=1)

        # Plot Bars
        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths - offsets
            rects = ax.barh(labels, widths, left=starts, height=0.9,
                            label=colname, color=color, zorder=2)

            for j, (w,s,o) in enumerate(zip(widths, starts, offsets)):
                if w > 0:
                    ax.text(s+(w/2.), j, w, horizontalalignment='center', verticalalignment='center', fontsize=9)
        
        # X Axis
        ax.set_xlim(-52, 52)
        ax.set_xticks(np.arange(-50, 51, 10))
        ax.xaxis.set_major_formatter(lambda x, pos: str(abs(int(x))))

        # Y Axis
        ax.invert_yaxis()
        ax.set_xlabel('Number of participants', fontsize=8)
        # ax.yaxis.set_ticks_position('none') 
        
        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_visible(False)
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], ncol=1, bbox_to_anchor=(0.1, 0.5),
                loc='center left', fontsize='x-small')

        # Set Background Color
        fig.set_facecolor('#FFFFFF')

        return fig, ax


    fig, ax = survey(results, category_names)

    fig.tight_layout()

    # plt.show()
    plt.savefig(join(RESULTS_DIR, 'questionnaire_responses.pdf'))


def main():
    
    # Load experiment logs
    # Return a database that contains all experiments organized into ind1, ind2, dir1, dir2 and send, no_send
    # e.g. stats['ind1']['send'] - all data from 'ind1' with 'send' condition
     
    stats = load_logs()
    # pprint.pprint(stats)

    # Load questionnaire results as a DataFrame
    q_df = load_questionnaires()
    print(q_df)

    # plot_points_scored(stats, include_partial_points=True)

    # plot_distance_traveled(stats)

    # plot_average_connectors(stats)

    # plot_distance_between_teams(stats)

    # plot_distance_between_teams_timeline(stats)

    # plot_distance_between_teams_vs_points(stats, include_partial_points=True)

    # plot_robots_shared(stats)

    # plot_robots_shared_vs_points(stats, include_partial_points=True)

    # plot_traveler_time(stats)

    # plot_traveler_time_vs_points(stats, include_partial_points=True)

    # plot_traveler_distance(stats)

    # plot_average_traveler_distance(stats)

    # plot_traveler_distance_vs_points(stats, include_partial_points=True)

    # plot_task_waiting_time_vs_points(stats, include_partial_points=True)

    # plot_learning_effect(stats, include_partial_points=True)

    # plot_preliminary(q_df)

    # plot_gaming_frequency_vs_points(stats, q_df, include_partial_points=True)

    # plot_task_load(q_df)

    # plot_global_task_load(q_df)

    # plot_global_task_load_vs_points(stats, q_df, include_partial_points=True)

    # plot_situational_awareness(q_df)

    # plot_global_situational_awareness(q_df)

    # plot_teammate_understanding(q_df)

    # plot_robot_understanding(q_df)

    plot_questionnaire_responses(q_df)


if __name__ == "__main__":
    main()
