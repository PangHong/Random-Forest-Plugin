/*
 * Copyright © 2016 Cask Data, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package co.cask.hydrator.plugin;

import co.cask.cdap.api.annotation.Description;
import co.cask.cdap.api.annotation.Name;
import co.cask.cdap.api.annotation.Plugin;
import co.cask.cdap.api.data.schema.Schema;
import co.cask.cdap.etl.api.batch.SparkSink;
import co.cask.hydrator.common.spark.SparkUtils;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import java.util.Map;
import javax.annotation.Nullable;

/**
 * Spark Sink plugin that trains a model based upon a label in the structured record using Random Forest Regression.
 * Writes this model to a FileSet.
 */
@Plugin(type = SparkSink.PLUGIN_TYPE)
@Name(RandomForestTrainer.PLUGIN_NAME)
@Description("Trains a regression model based upon a particular label and features of a record.")
public class RandomForestTrainer extends SparkMLTrainer {
  public static final String PLUGIN_NAME = "RandomForestTrainer";
  //Impurity measure of the homogeneity of the labels at the node. Expected value for regression is "variance".
  private static final String IMPURITY = "variance";
  private static final String FEATURESUBSETSTRATEGY = "auto"; // let the algorithm choose feature subset strategy
  private RandomForestTrainerConfig config;

  public RandomForestTrainer(RandomForestTrainerConfig config) {
    super(config);
    this.config = config;
  }

  @Override
  public void trainModel(SparkContext context, Schema schema, JavaRDD<LabeledPoint> trainingData, String outputPath) {
    Map<Integer, Integer> categoricalFeaturesInfo =
      SparkUtils.getCategoricalFeatureInfo(schema, config.featureFieldsToInclude, config.featureFieldsToExclude,
                                           config.labelField, config.cardinalityMapping);

    final RandomForestModel model = RandomForest.trainRegressor(trainingData,
 categoricalFeaturesInfo, 
 config.numTrees,
 FEATURESUBSETSTRATEGY,
 IMPURITY, 
 config.maxDepth,
 config.maxBins,
 config.seed);

    model.save(context, outputPath);
  }

  /**
   * Configuration for RandomForestTrainer.
   */
  public static class RandomForestTrainerConfig extends MLTrainerConfig {

    @Nullable
    @Description("List of the categorical features along with the maximum number of unique values that feature can " +
      "exist in. This is a comma-separated list of key-value pairs, where each pair is separated by a colon ':' and " +
      "specifies the feature and its cardinality. For example, 'daysOfTheWeek:7', this indicates that the feature " +
      "'daysOfTheWeek' is categorical with '7' categories indexed from 0: {0, 1, 2, 3, 4, 5, 6}.")
    private String cardinalityMapping;

    @Nullable
    @Description("Number of the tree.\n Use more in practice Default is 3.")
    private Integer numTrees;


    @Nullable
    @Description("Maximum depth of the tree.\n For example, depth 0 means 1 leaf node; depth 1 means 1 internal node " +
      "+ 2 leaf nodes. Default is 10.")
    private Integer maxDepth;

    @Nullable
    @Description("Maximum number of bins used for splitting when discretizing continuous features. " +
      "DecisionTree requires maxBins to be at least as large as the number of values in each categorical feature. " +
      "Default is 100.")
    private Integer maxBins;

    @Nullable
    @Description("set random seed.\nDefault is 12345.")
    private Integer seed;

    public RandomForestTrainerConfig() {
      super();
      numTrees = 3;
      maxDepth = 10;
      maxBins = 100;
      seed = 12345;
    }

    public RandomForestTrainerConfig(
 String fileSetName,
 @Nullable String path,
 @Nullable String featuresToInclude,
 @Nullable String featuresToExclude,
 @Nullable String cardinalityMapping,
 String labelField,
 @Nullable Integer numTrees,
 @Nullable Integer maxDepth,
 @Nullable Integer maxBins,
 @Nullable Integer seed) {
      super(fileSetName, path, featuresToInclude, featuresToExclude, labelField);
      this.cardinalityMapping = cardinalityMapping;
      this.numTrees = numTrees;
      this.maxDepth = maxDepth;
      this.maxBins = maxBins;
      this.seed = seed;
    }
  }
}
