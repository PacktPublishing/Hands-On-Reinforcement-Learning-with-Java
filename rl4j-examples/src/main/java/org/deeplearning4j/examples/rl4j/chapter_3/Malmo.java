package org.deeplearning4j.examples.rl4j.chapter_3;

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.util.DataManager;

import org.deeplearning4j.malmo.MalmoBox;
import org.deeplearning4j.malmo.MalmoActionSpaceDiscrete;
import org.deeplearning4j.malmo.MalmoConnectionError;
import org.deeplearning4j.malmo.MalmoDescretePositionPolicy;
import org.deeplearning4j.malmo.MalmoEnv;
import org.deeplearning4j.malmo.MalmoObservationSpace;
import org.deeplearning4j.malmo.MalmoObservationSpacePosition;
import org.nd4j.linalg.learning.config.Adam;

import java.io.IOException;
import java.util.logging.Logger;

/**
 * step 1 - start https://github.com/microsoft/malmo -
 * https://github.com/Microsoft/malmo/blob/master/scripts/python-wheel/README.md
 * step 2 - ./Minecraft/launchClient.sh
 * step 3 - change CLIFF_WALING_XML_CONFIG_PATH to you path
 * step 4 - python3 Python_Examples/run_mission.py
 * step 5 - set $MALMO_HOME - for example to: /Users/tomaszlelek/Downloads/Malmo-0.37.0-Mac-64bit_withBoost_Python3.6
 */
public class Malmo {
    //set it to your local path
    public static final String CLIFF_WALING_XML_CONFIG_PATH = "/Users/tomaszlelek/IntelliJ_workspace/dl4j-examples/rl4j-examples/cliff_walking_rl4j.xml";

    private static QLearning.QLConfiguration MALMO_QL = new QLearning.QLConfiguration(123, //Random seed
        200, //Max step By epoch
        200000, //Max step
        200000, //Max size of experience replay
        32, //size of batches
        50, //target update (hard)
        10, //num step noop warmup
        0.01, //reward scaling
        0.99, //gamma
        1.0, //td-error clipping
        0.15f, //min epsilon
        1000, //num step for eps greedy anneal
        true //double DQN
    );

    public static DQNFactoryStdDense.Configuration MALMO_NET = DQNFactoryStdDense.Configuration.builder().l2(0.00)
        .updater(new Adam(0.001)).numHiddenNodes(50).numLayer(3).build();

    public static void main(String[] args) throws IOException {

        try {
            malmoCliffWalk();
            loadMalmoCliffWalk();
        } catch (MalmoConnectionError e) {
            System.out.println(
                "To run this example, download and start Project Malmo found at https://github.com/Microsoft/malmo.");
        }
    }

    private static MalmoEnv createMDP() {
        MalmoActionSpaceDiscrete possibleActions =
            new MalmoActionSpaceDiscrete("movenorth 1", "movesouth 1", "movewest 1", "moveeast 1");
        possibleActions.setRandomSeed(123);

        MalmoObservationSpace observationSpace = createObservationXYZSpace();


        MalmoDescretePositionPolicy captureRewardPolicy = new MalmoDescretePositionPolicy();

        return new MalmoEnv(
            CLIFF_WALING_XML_CONFIG_PATH, possibleActions, observationSpace, captureRewardPolicy);
    }

    private static MalmoObservationSpace createObservationXYZSpace() {
        return new MalmoObservationSpacePosition();
    }

    public static void malmoCliffWalk() throws MalmoConnectionError, IOException {
        DataManager manager = new DataManager(true);

        MalmoEnv mdp = createMDP();

        QLearningDiscreteDense<MalmoBox> dql = new QLearningDiscreteDense<MalmoBox>(mdp, MALMO_NET, MALMO_QL, manager);

        dql.train();

        DQNPolicy<MalmoBox> pol = dql.getPolicy();

        pol.save("cliffwalk.policy");

        mdp.close();
    }

    // showcase serialization by using the trained agent on a new similar mdp
    private static void loadMalmoCliffWalk() throws MalmoConnectionError, IOException {
        MalmoEnv mdp = createMDP();

        DQNPolicy<MalmoBox> pol = loadPreviousPolicy();

        double rewards = evaluateAgent(mdp, pol);

        mdp.close();


        printAverageRewards(rewards);
    }

    private static void printAverageRewards(double rewards) {
        Logger.getAnonymousLogger().info("average: " + rewards / 10);
    }

    private static DQNPolicy<MalmoBox> loadPreviousPolicy() throws IOException {
        return DQNPolicy.load("cliffwalk.policy");
    }

    private static double evaluateAgent(MalmoEnv mdp, DQNPolicy<MalmoBox> pol) {
        double rewards = 0;
        for (int i = 0; i < 10; i++) {
            double reward = pol.play(mdp);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }
        return rewards;
    }
}
