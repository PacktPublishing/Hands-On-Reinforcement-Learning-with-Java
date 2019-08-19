package org.deeplearning4j.examples.rl4j.chapter_2;

import java.io.IOException;
import org.deeplearning4j.rl4j.space.Box;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.gym.GymEnv;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;

import java.util.logging.Logger;


// Using https://github.com/openai/gym-http-api
// python gym_http_server.py
public class Cartpole
{

    private static QLearning.QLConfiguration CARTPOLE_QL =
            new QLearning.QLConfiguration(
                    123,
                    10,
                    150000,
                    150000,
                    32,
                    500,
                    10,
                    0.01,
                    0.99,
                    1.0,
                    0.1f,
                    1000,
                    true
            );

    private static DQNFactoryStdDense.Configuration CARTPOLE_NET =
        DQNFactoryStdDense.Configuration.builder()
            .l2(0.001).updater(new Adam(0.0005))
            .numHiddenNodes(16).numLayer(3).build();

    public static void main(String[] args) throws IOException {
        cartPole();
        loadCartpole();
    }

    private static void cartPole() throws IOException {

        DataManager manager = recordTrainDataAndSaveInNewFolder();

        GymEnv<Box, Integer, DiscreteSpace> mdp = defineMdpFromGym();

        QLearningDiscreteDense<Box> dql = defineTraining(manager, mdp);

        dql.train();

        DQNPolicy<Box> pol = dql.getPolicy();

        saveForFutureReuse(pol);

        mdp.close();


    }

    private static void saveForFutureReuse(DQNPolicy<Box> pol) throws IOException {
        pol.save("/tmp/pol1");
    }

    private static QLearningDiscreteDense<Box> defineTraining(DataManager manager, GymEnv<Box, Integer, DiscreteSpace> mdp) {
        return new QLearningDiscreteDense<>(mdp, CARTPOLE_NET, CARTPOLE_QL, manager);
    }

    private static GymEnv<Box, Integer, DiscreteSpace> defineMdpFromGym() {
        GymEnv<Box, Integer, DiscreteSpace> mdp = null;
        try {
            mdp = new GymEnv<>("CartPole-v0", true, true);
        } catch (RuntimeException e){
            System.out.print("To run this example, download and start the gym-http-api repo found at https://github.com/openai/gym-http-api.");
        }
        return mdp;
    }

    private static DataManager recordTrainDataAndSaveInNewFolder() throws IOException {
        return new DataManager(true);
    }


    private static void loadCartpole() throws IOException {
        GymEnv mdp2 = loadGymEnv();

        DQNPolicy<Box> pol2 = loadPreviousAgent();

        double rewards = evaluateResults(mdp2, pol2);

        Logger.getAnonymousLogger().info("average: " + rewards/1000);
    }

    private static double evaluateResults(GymEnv mdp2, DQNPolicy<Box> pol2) {
        double rewards = 0;
        for (int i = 0; i < 10; i++) {
            double reward = pol2.play(mdp2);
            rewards += reward;
            Logger.getAnonymousLogger().info("Reward: " + reward);
        }
        return rewards;
    }

    private static DQNPolicy<Box> loadPreviousAgent() throws IOException {
        return DQNPolicy.load("/tmp/pol1");
    }

    private static GymEnv loadGymEnv() {
        return new GymEnv("CartPole-v0", true, false);
    }
}
