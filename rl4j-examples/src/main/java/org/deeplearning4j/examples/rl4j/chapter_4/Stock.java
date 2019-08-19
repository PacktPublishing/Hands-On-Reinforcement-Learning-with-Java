package org.deeplearning4j.examples.rl4j.chapter_4;

import java.io.IOException;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.Learning;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscrete;
import org.deeplearning4j.rl4j.learning.async.nstep.discrete.AsyncNStepQLearningDiscreteDense;
import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.mdp.toy.HardDeteministicToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToy;
import org.deeplearning4j.rl4j.mdp.toy.SimpleToyState;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;


/**
 * Stock prediction using Toy abstraction.
 * Stock state is a toy state. Based on our action (buy/sell) the toy/stock state can change -
 * The actual value of Stock can go up or down.
 */
public class Stock {


    public static QLearning.QLConfiguration STOCK_QL =
            new QLearning.QLConfiguration(
                    123,   //Random seed
                    100000,//Max step By epoch
                    80000, //Max step
                    10000, //Max size of experience replay
                    32,    //size of batches
                    100,   //target update (hard)
                    0,     //num step noop warmup
                    0.05,  //reward scaling
                    0.99,  //gamma
                    10.0,  //td-error clipping
                    0.1f,  //min epsilon
                    2000,  //num step for eps greedy anneal
                    true   //double DQN
            );


    public static AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration STOCK_ASYNC_MULTI_THREAD_QL =
            new AsyncNStepQLearningDiscrete.AsyncNStepQLConfiguration(
                    123,        //Random seed
                    100000,     //Max step By epoch
                    80000,      //Max step
                    8,          //Number of threads
                    5,          //t_max
                    100,        //target update (hard)
                    0,          //num step noop warmup
                    0.1,        //reward scaling
                    0.99,       //gamma
                    10.0,       //td-error clipping
                    0.1f,       //min epsilon
                    2000        //num step for eps greedy anneal
            );


    public static DQNFactoryStdDense.Configuration STOCK_NET =
             DQNFactoryStdDense.Configuration.builder()
        .l2(0.01).updater(new Adam(1e-2)).numLayer(3).numHiddenNodes(16).build();

    public static void main(String[] args) throws IOException {
        simpleStock();
        //stockAsyncMultiThread();

    }

    public static void simpleStock() throws IOException {

        DataManager manager = getDataManager();

        SimpleToy mdp = createStockAbstraction();

        Learning<SimpleToyState, Integer, DiscreteSpace, IDQN> dql = new QLearningDiscreteDense<>(mdp, STOCK_NET, STOCK_QL, manager);

        mdp.setFetchable(dql);

        dql.train();

        mdp.close();

    }

    private static SimpleToy createStockAbstraction() {
        return new SimpleToy(20);
    }

    private static DataManager getDataManager() throws IOException {
        return new DataManager();
    }


    public static void stockAsyncMultiThread() throws IOException {

        DataManager manager = getDataManager();

        SimpleToy mdp = createStockAbstraction();

        AsyncNStepQLearningDiscreteDense dql = new AsyncNStepQLearningDiscreteDense<>(mdp, STOCK_NET, STOCK_ASYNC_MULTI_THREAD_QL, manager);

        mdp.setFetchable(dql);

        dql.train();

        mdp.close();

    }

}
