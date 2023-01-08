import numpy as np
import NeuralThompsonSampling as nts
import env
from agents import CascadingBanditTS, CascadingBanditKLUCB, CascadingBanditUCB1, CascadingBanditEpsilonGreedy
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms #for confidence intervals
from joblib import Parallel, delayed
import time
import pickle


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def msePrime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def experimentsParallel(numTries=1, steps=200, numArms=50, numProposed=10, numLayers=1, width=10, ntss=True, numOpt=100, lr=1e-2, hist=100, bandit=None, best_optimism=0.1, true_a0 = 1, true_b0=40):
    bColors = ['m', 'y', 'c']
    banditName = ["CascadeTS", "UCB", "UCB Optimism"]

    items = np.eye(numArms, numArms).reshape(numArms, 1, numArms)

    errsNTSS = np.zeros((numTries, steps))
    errsNTS = np.zeros((numTries, steps))
    errsNN = np.zeros((numTries, steps))
    errsNNT = np.zeros((numTries, steps))
    errsBandits = np.zeros((len(bColors), numTries, steps))

    for k in range(numTries):
        if ntss:
            #neralTS Structured
            netTSS = nts.ThompsonNetwork(numArms, width, numLayers)
            errNTSS = 0
            bandit.round_success = []
            bandit.round_failure = []
            for j in range(steps):
                #sample rewards
                predicts = np.apply_along_axis(netTSS.sampleReward, 0, items).reshape(numArms)
                #select best arms
                armsPlayed = (-predicts).argsort()[:numProposed]
                if j != steps-1:
                    #play best arms
                    rew = bandit.get_stochastic_reward(armsPlayed)
                    obs = bandit.get_observation()
                    #create arm reward list
                    if len(obs["round_success"]) > 0:
                        obsList = np.array([[success, 1] for success in obs["round_success"]])
                        if len(obs["round_failure"]) > 0:
                            obsList = np.concatenate(
                                (
                                    np.array([[fail, 0] for fail in obs["round_failure"]]),
                                    obsList
                                ), 0)
                    else:
                        obsList = np.array([[fail, 0] for fail in obs["round_failure"]])
                    netTSS.history = np.concatenate((netTSS.history, obsList), 0)[-hist:]
                    netTSS.updateTheta(numOpt, lr)
                    for arm in items[armsPlayed]:
                        netTSS.updateU(arm)
                #get regret
                optimal_reward = bandit.get_optimal_reward()
                rew = bandit.get_expected_reward(armsPlayed)
                errNTSS += optimal_reward - rew
                errsNTSS[k, j] = errNTSS
            print(f"NeuralThompson Structured: {errNTSS}")


            #neuralTS
            netTS = nts.ThompsonNetwork(numArms, width, numLayers, False)

            errNTS = 0
            bandit.round_success = []
            bandit.round_failure = []
            for j in range(steps):
                #sample rewards
                predicts = np.apply_along_axis(netTS.sampleReward, 0, items).reshape(numArms)
                #select best arms
                armsPlayed = (-predicts).argsort()[:numProposed]
                if j != steps-1:
                    #play best arms
                    rew = bandit.get_stochastic_reward(armsPlayed)
                    obs = bandit.get_observation()
                    #create arm reward list
                    if len(obs["round_success"]) > 0:
                        obsList = np.array([[success, 1] for success in obs["round_success"]])
                        if len(obs["round_failure"]) > 0:
                            obsList = np.concatenate(
                                (
                                    np.array([[fail, 0] for fail in obs["round_failure"]]),
                                    obsList
                                ), 0)
                    else:
                        obsList = np.array([[fail, 0] for fail in obs["round_failure"]])
                    netTS.history = np.concatenate((netTS.history, obsList), 0)[-hist:]
                    netTS.updateTheta(numOpt, lr)
                    for arm in items[armsPlayed]:
                        netTS.updateU(arm)
                #get regret
                optimal_reward = bandit.get_optimal_reward()
                rew = bandit.get_expected_reward(armsPlayed)
                errNTS += optimal_reward - rew
                errsNTS[k, j] = errNTS
            print(f"NeuralThompson: {errNTS}")

        #general net
        net = nts.FullyConnectedNeuralNetwork(numArms, width, numLayers, True, mse, msePrime)

        errNN = 0
        bandit.round_success = []
        bandit.round_failure = []
        for j in range(steps):
            predicts = np.apply_along_axis(net.predict, 0, items).reshape(numArms)
            armsPlayed = (-predicts).argsort()[:numProposed]
            rew = bandit.get_stochastic_reward(armsPlayed)
            obs = bandit.get_observation()
            for fail in obs["round_failure"]:
                net.updateParams(items[fail], np.array([[0]]), 0.01)
            for success in obs["round_success"]:
                net.updateParams(items[success], np.array([[1]]), 0.01)
            optimal_reward = bandit.get_optimal_reward()
            #regret is optimal against expected reward
            rew = bandit.get_expected_reward(armsPlayed)
            errNN += optimal_reward-rew
            errsNN[k, j] = errNN

        print(f"NeuralNetwork {errNN}")


        #generalNet more training
        netTraining = nts.FullyConnectedNeuralNetwork(numArms, width, numLayers, True, mse, msePrime)

        errNNT = 0
        bandit.round_success = []
        bandit.round_failure = []
        for j in range(steps):
            predicts = np.apply_along_axis(netTraining.predict, 0, items).reshape(numArms)
            armsPlayed = (-predicts).argsort()[:numProposed]
            rew = bandit.get_stochastic_reward(armsPlayed)
            obs = bandit.get_observation()
            if len(obs["round_success"]) > 0:
                obsList = np.array([[success, 1] for success in obs["round_success"]])
                if len(obs["round_failure"]) > 0:
                    obsList = np.concatenate(
                        (
                            np.array([[fail, 0] for fail in obs["round_failure"]]),
                            obsList
                        ), 0)
            else:
                obsList = np.array([[fail, 0] for fail in obs["round_failure"]])
            netTraining.history = np.concatenate((netTraining.history, obsList), 0)[-1000:]
            for arm, rew in netTraining.history:
                netTraining.updateParams(items[arm], np.array([[rew]]), 0.01)
            optimal_reward = bandit.get_optimal_reward()
            rew = bandit.get_expected_reward(armsPlayed)
            errNNT += optimal_reward - rew
            errsNNT[k, j] = errNNT

        print(f"NeuralNetwork better Training {errNNT}")

        #stuff from given code
        agent = CascadingBanditTS(numArms, numProposed, a0=true_a0, b0=true_b0)
        agentUCB = CascadingBanditUCB1(numArms, numProposed, a0=true_a0, b0=true_b0, optimism=1)
        agentUCBGood = CascadingBanditUCB1(numArms, numProposed, a0=true_a0, b0=true_b0, optimism=best_optimism)
        agents = [agent, agentUCB, agentUCBGood]


        for j in range(len(agents)):
            bandit.round_success = []
            bandit.round_failure = []
            cum_regret = 0
            for i in range(steps):
                observation = bandit.get_observation()
                action = agent.pick_action(observation)

                # Compute useful stuff for regret calculations
                optimal_reward = bandit.get_optimal_reward()
                expected_reward = bandit.get_expected_reward(action)
                reward = bandit.get_stochastic_reward(action)

                # Update the agent using realized rewards + bandit learing
                agent.update_observation(observation, action, reward)

                # Log whatever we need for the plots we will want to use.
                instant_regret = optimal_reward - expected_reward
                cum_regret += instant_regret
                errsBandits[j, k, i] = cum_regret
            print(f"{banditName[j]}: {cum_regret}")
    return dict({
        "neuralTS_structure": errsNTSS,
        "neuralTS": errsNTS,
        "nn": errsNN,
        "nn_train": errsNNT,
        "bandits": errsBandits,
    })

def experiments(numTries=20, steps=200, numArms=50, numProposed=10, numLayers=1, width=10, best_optimism=0.1, true_a0=1, true_b0=40):
    bColors = ['m', 'y', 'c']
    banditName = ["CascadeTS", "UCB", "UCB Optimism"]

    items = np.eye(numArms, numArms).reshape(numArms, 1, numArms)

    errsNTSS = np.zeros((numTries, steps))
    errsNTS = np.zeros((numTries, steps))
    errsNN = np.zeros((numTries, steps))
    errsNNT = np.zeros((numTries, steps))
    errsBandits = np.zeros((len(bColors), numTries, steps))

    bandit = env.CascadingBandit(numArms, numProposed, true_a0, true_b0)
    for k in range(numTries):
        print(f"round {k+1}")
        #neralTS Structured
        netTSS = nts.ThompsonNetwork(numArms, width, numLayers)
        errNTSS = 0
        bandit.round_success = []
        bandit.round_failure = []
        for j in range(steps):
            #sample rewards
            predicts = np.apply_along_axis(netTSS.sampleReward, 0, items).reshape(numArms)
            #select best arms
            armsPlayed = (-predicts).argsort()[:numProposed]
            if j != steps-1:
                #play best arms
                rew = bandit.get_stochastic_reward(armsPlayed)
                obs = bandit.get_observation()
                #create arm reward list
                if len(obs["round_success"]) > 0:
                    obsList = np.array([[success, 1] for success in obs["round_success"]])
                    if len(obs["round_failure"]) > 0:
                        obsList = np.concatenate(
                            (
                                np.array([[fail, 0] for fail in obs["round_failure"]]),
                                obsList
                            ), 0)
                else:
                    obsList = np.array([[fail, 0] for fail in obs["round_failure"]])
                netTSS.history = np.concatenate((netTSS.history, obsList), 0)[-100:]
                netTSS.updateTheta(100)
                for arm in items[armsPlayed]:
                    netTSS.updateU(arm)
            #get regret
            optimal_reward = bandit.get_optimal_reward()
            rew = bandit.get_expected_reward(armsPlayed)
            errNTSS += optimal_reward - rew
            errsNTSS[k, j] = errNTSS
        print(f"NeuralThompson Structured: {errNTSS}")


        #neuralTS
        netTS = nts.ThompsonNetwork(numArms, width, numLayers, False)

        errNTS = 0
        bandit.round_success = []
        bandit.round_failure = []
        for j in range(steps):
            #sample rewards
            predicts = np.apply_along_axis(netTS.sampleReward, 0, items).reshape(numArms)
            #select best arms
            armsPlayed = (-predicts).argsort()[:numProposed]
            if j != steps-1:
                #play best arms
                rew = bandit.get_stochastic_reward(armsPlayed)
                obs = bandit.get_observation()
                #create arm reward list
                if len(obs["round_success"]) > 0:
                    obsList = np.array([[success, 1] for success in obs["round_success"]])
                    if len(obs["round_failure"]) > 0:
                        obsList = np.concatenate(
                            (
                                np.array([[fail, 0] for fail in obs["round_failure"]]),
                                obsList
                            ), 0)
                else:
                    obsList = np.array([[fail, 0] for fail in obs["round_failure"]])
                netTS.history = np.concatenate((netTS.history, obsList), 0)[-100:]
                netTS.updateTheta(50, 0.1)
                for arm in items[armsPlayed]:
                    netTS.updateU(arm)
            #get regret
            optimal_reward = bandit.get_optimal_reward()
            rew = bandit.get_expected_reward(armsPlayed)
            errNTS += optimal_reward - rew
            errsNTS[k, j] = errNTS
        print(f"NeuralThompson: {errNTS}")

        #general net
        net = nts.FullyConnectedNeuralNetwork(numArms, width, numLayers, True, mse, msePrime)

        errNN = 0
        bandit.round_success = []
        bandit.round_failure = []
        for j in range(steps):
            predicts = np.apply_along_axis(net.predict, 0, items).reshape(numArms)
            armsPlayed = (-predicts).argsort()[:numProposed]
            rew = bandit.get_stochastic_reward(armsPlayed)
            obs = bandit.get_observation()
            for fail in obs["round_failure"]:
                net.updateParams(items[fail], np.array([[0]]), 0.01)
            for success in obs["round_success"]:
                net.updateParams(items[success], np.array([[1]]), 0.01)
            optimal_reward = bandit.get_optimal_reward()
            #regret is optimal against expected reward
            rew = bandit.get_expected_reward(armsPlayed)
            errNN += optimal_reward-rew
            errsNN[k, j] = errNN

        print(f"NeuralNetwork {errNN}")


        #generalNet more training
        netTraining = nts.FullyConnectedNeuralNetwork(numArms, width, numLayers, True, mse, msePrime)

        errNNT = 0
        bandit.round_success = []
        bandit.round_failure = []
        for j in range(steps):
            predicts = np.apply_along_axis(netTraining.predict, 0, items).reshape(numArms)
            armsPlayed = (-predicts).argsort()[:numProposed]
            rew = bandit.get_stochastic_reward(armsPlayed)
            obs = bandit.get_observation()
            if len(obs["round_success"]) > 0:
                obsList = np.array([[success, 1] for success in obs["round_success"]])
                if len(obs["round_failure"]) > 0:
                    obsList = np.concatenate(
                        (
                            np.array([[fail, 0] for fail in obs["round_failure"]]),
                            obsList
                        ), 0)
            else:
                obsList = np.array([[fail, 0] for fail in obs["round_failure"]])
            netTraining.history = np.concatenate((netTraining.history, obsList), 0)[-1000:]
            for arm, rew in netTraining.history:
                netTraining.updateParams(items[arm], np.array([[rew]]), 0.01)
            optimal_reward = bandit.get_optimal_reward()
            rew = bandit.get_expected_reward(armsPlayed)
            errNNT += optimal_reward - rew
            errsNNT[k, j] = errNNT

        print(f"NeuralNetwork better Training {errNNT}")

        #stuff from given code
        agent = CascadingBanditTS(numArms, numProposed, a0=true_a0, b0=true_b0)
        agentUCB = CascadingBanditUCB1(numArms, numProposed, a0=true_a0, b0=true_b0, optimism=1)
        agentUCBGood = CascadingBanditUCB1(numArms, numProposed, a0=true_a0, b0=true_b0, optimism=best_optimism)
        agents = [agent, agentUCB, agentUCBGood]


        for j in range(len(agents)):
            bandit.round_success = []
            bandit.round_failure = []
            cum_regret = 0
            for i in range(steps):
                observation = bandit.get_observation()
                action = agent.pick_action(observation)

                # Compute useful stuff for regret calculations
                optimal_reward = bandit.get_optimal_reward()
                expected_reward = bandit.get_expected_reward(action)
                reward = bandit.get_stochastic_reward(action)

                # Update the agent using realized rewards + bandit learing
                agent.update_observation(observation, action, reward)

                # Log whatever we need for the plots we will want to use.
                instant_regret = optimal_reward - expected_reward
                cum_regret += instant_regret
                errsBandits[j, k, i] = cum_regret
            print(f"{banditName[j]}: {cum_regret}")

    # plot NTSS
    mean = np.mean(errsNTSS, axis=0)
    ci = np.apply_along_axis(lambda a: sms.DescrStatsW(a).tconfint_mean(0.2), 0, errsNTSS)
    plt.plot(mean, label=f"NTS structure width {netTSS.width}, layers {netTSS.numLayers+2}", color='b')
    plt.fill_between(range(steps), ci[0, ], ci[1, ], color='b', alpha=.1)

    # NTS
    mean = np.mean(errsNTS, axis=0)
    ci = np.apply_along_axis(lambda a: sms.DescrStatsW(a).tconfint_mean(0.2), 0, errsNTS)
    plt.plot(mean, label=f"NTS width {netTS.width}, layers {netTS.numLayers+2}", color='r')
    plt.fill_between(range(steps), ci[0, ], ci[1, ], color='r', alpha=.1)

    #NN
    mean = np.mean(errsNN, axis=0)
    ci = np.apply_along_axis(lambda a: sms.DescrStatsW(a).tconfint_mean(0.2), 0, errsNN)
    plt.plot(mean, label=f"NN width {net.width}, layers {net.numLayers+2}", color='g')
    plt.fill_between(range(steps), ci[0, ], ci[1, ], color='g', alpha=.1)


    #NN Training
    mean = np.mean(errsNNT, axis=0)
    ci = np.apply_along_axis(lambda a: sms.DescrStatsW(a).tconfint_mean(0.2), 0, errsNNT)
    plt.plot(mean, label=f"NN longer train width {netTraining.width}, layers {netTraining.numLayers+2}", color='lightgreen')
    plt.fill_between(range(steps), ci[0, ], ci[1, ], color='lightgreen', alpha=.1)

    #given stuff
    for j in range(len(agents)):
        mean = np.mean(errsBandits[j], axis=0)
        ci = np.apply_along_axis(lambda a: sms.DescrStatsW(a).tconfint_mean(0.2), 0, errsBandits[j])
        plt.plot(mean, label=f"{banditName[j]}", color=bColors[j])
        plt.fill_between(range(steps), ci[0, ], ci[1, ], color=bColors[j], alpha=.1)

    # show the results
    plt.xlabel("Iterations")
    plt.ylabel("Average total Regret")
    plt.title(f"Cascading Bandits with Optimal reward: {bandit.get_optimal_reward()}")
    plt.legend()
    plt.savefig(f"experiments_{numTries}_{steps}_{numArms}_{numProposed}_{best_optimism}_{numLayers}_{width}_{true_a0}_{true_b0}.svg", format="svg")
    plt.show()

def combineParaRes(res, numTries, steps, numAgents, ntss):
    if ntss:
        errsNTSS = np.zeros((numTries, steps))
        errsNTS = np.zeros((numTries, steps))
    else:
        errsNTSS = None
        errsNTS = None
    errsNN = np.zeros((numTries, steps))
    errsNNT = np.zeros((numTries, steps))
    errsBandits = np.zeros((numTries, numAgents, steps))

    for i, run in enumerate(res):
        if ntss:
            errsNTSS[i] = res[i]["neuralTS_structure"]
            errsNTS[i] = res[i]["neuralTS"]
        errsNN[i] = res[i]["nn"]
        errsNNT[i] = res[i]["nn_train"]
        errsBandits[i] = res[i]["bandits"].reshape(numAgents, steps)

    return errsNTSS, errsNTS, errsNN, errsNNT, errsBandits


def fullExp(numTries=30, steps=200, numArms=50, numProposed=10, numLayers=1, width=10, ntss=True, numOps=100, lr=1e-2, hist=100, best_optimism=0.1, true_a0=1, true_b0=40):
    # init
    bColors = ['m', 'y', 'c']
    banditName = ["CascadeTS", "UCB", "UCB Optimism"]

    items = np.eye(numArms, numArms).reshape(numArms, 1, numArms)

    bandit = env.CascadingBandit(numArms, numProposed, true_a0, true_b0)

    res = Parallel(n_jobs=numTries)(delayed(experimentsParallel)(1, steps, numArms, numProposed, numLayers, width, ntss, numOps, lr, hist, bandit, best_optimism, true_a0, true_b0) for i in range(numTries))

    errsNTSS, errsNTS, errsNN, errsNNT, errsBandits = combineParaRes(res, numTries, steps, len(bColors), ntss)

    plt.figure()

    # plot NTSS
    mean = np.mean(errsNTSS, axis=0)
    ci = np.apply_along_axis(lambda a: sms.DescrStatsW(a).tconfint_mean(0.2), 0, errsNTSS)
    plt.plot(mean, label=f"NTS structure width {width}, layers {numLayers + 2}", color='b')
    plt.fill_between(range(steps), ci[0,], ci[1,], color='b', alpha=.1)

    # NTS
    mean = np.mean(errsNTS, axis=0)
    ci = np.apply_along_axis(lambda a: sms.DescrStatsW(a).tconfint_mean(0.2), 0, errsNTS)
    plt.plot(mean, label=f"NTS width {width}, layers {numLayers + 2}", color='r')
    plt.fill_between(range(steps), ci[0,], ci[1,], color='r', alpha=.1)

    # NN
    mean = np.mean(errsNN, axis=0)
    ci = np.apply_along_axis(lambda a: sms.DescrStatsW(a).tconfint_mean(0.2), 0, errsNN)
    plt.plot(mean, label=f"NN width {width}, layers {numLayers + 2}", color='g')
    plt.fill_between(range(steps), ci[0,], ci[1,], color='g', alpha=.1)

    # NN Training
    mean = np.mean(errsNNT, axis=0)
    ci = np.apply_along_axis(lambda a: sms.DescrStatsW(a).tconfint_mean(0.2), 0, errsNNT)
    plt.plot(mean, label=f"NN longer train width {width}, layers {numLayers + 2}",
             color='lightgreen')
    plt.fill_between(range(steps), ci[0, ], ci[1, ], color='lightgreen', alpha=.1)

    # given stuff
    for j in range(len(bColors)):
        mean = np.mean(errsBandits[:, j, :], axis=0)
        ci = np.apply_along_axis(lambda a: sms.DescrStatsW(a).tconfint_mean(0.2), 0, errsBandits[:, j, :])
        plt.plot(mean, label=f"{banditName[j]}", color=bColors[j])
        plt.fill_between(range(steps), ci[0, ], ci[1, ], color=bColors[j], alpha=.1)

    # show the results
    plt.xlabel("Iterations")
    plt.ylabel("Average total Regret")
    plt.title(f"Cascading Bandits with Optimal reward: {bandit.get_optimal_reward()}")
    plt.legend()
    plt.savefig(
        f"experiments_{numTries}_{steps}_{numArms}_{numProposed}_{best_optimism}_{numLayers}_{width}_{numOps}_{lr}_{hist}_{true_a0}_{true_b0}_{time.strftime('%Y%m%d_%H%M%S')}}.svg",
        format="svg")

    # create a binary pickle file
    f = open(f"data_{time.strftime('%Y%m%d_%H%M%S')}.pkl", "wb")

    # write the python object (dict) to pickle file
    pickle.dump(res, f)

    # close file
    f.close()
    plt.show()
    return errsNTSS, errsNTS, errsNN, errsNNT, errsBandits




if __name__ == "__main__":

    numTries = 50
    steps = 250
    arms = 50
    numProposed = 10
    numLayer = 2
    width = 10
    optSteps = 200
    lr = 5e-3

    fullExp(numTries, steps, arms, numProposed, numLayer, width, True, optSteps, lr)

    numTries = 50
    steps = 250
    arms = 50
    numProposed = 10
    numLayer = 1
    width = 25
    optSteps = 200
    lr = 5e-3

    fullExp(numTries, steps, arms, numProposed, numLayer, width, True, optSteps, lr)

    numTries = 50
    steps = 250
    arms = 50
    numProposed = 10
    numLayer = 3
    width = 6
    optSteps = 100
    lr = 1e-3

    fullExp(numTries, steps, arms, numProposed, numLayer, width, True, optSteps, lr, 300)

    numTries = 50
    steps = 250
    arms = 50
    numProposed = 10
    numLayer = 2
    width = 10
    optSteps = 100


    fullExp(numTries, steps, arms, numProposed, numLayer, width, True, optSteps, lr, 300)

    numTries = 50
    steps = 250
    arms = 50
    numProposed = 10
    numLayer = 1
    width = 25
    optSteps = 100
    lr = 5e-3

    fullExp(numTries, steps, arms, numProposed, numLayer, width, True, optSteps, lr, 300)

    numTries = 50
    steps = 250
    arms = 50
    numProposed = 10
    numLayer = 3
    width = 6
    optSteps = 50

    fullExp(numTries, steps, arms, numProposed, numLayer, width, True, optSteps, lr, 300)


    #no neural thompson anymore

    numTries = 50
    steps = 10000
    arms = 50
    numProposed = 10
    numLayer = 3
    width = 6


    fullExp(numTries, steps, arms, numProposed, numLayer, width, False)
    numTries = 50
    steps = 10000
    arms = 50
    numProposed = 10
    numLayer = 1
    width = 25


    fullExp(numTries, steps, arms, numProposed, numLayer, width, False)

    #no neural thompson anymore

    numTries = 50
    steps = 10000
    arms = 1000
    numProposed = 100
    numLayer = 3
    width = 6


    fullExp(numTries, steps, arms, numProposed, numLayer, width, False)
    numTries = 50
    steps = 10000
    arms = 1000
    numProposed = 100
    numLayer = 1
    width = 25


    fullExp(numTries, steps, arms, numProposed, numLayer, width, False)
