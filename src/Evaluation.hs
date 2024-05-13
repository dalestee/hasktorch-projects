module Evaluation (accuracy, precision, recall, extractBinary) where

import Torch.Layer.MLP    (mlpLayer, MLPParams)
import Torch.Tensor       (asTensor, asValue)

extractBinary :: [([Float], Float)] -> MLPParams -> (Int, Int, Int, Int)
extractBinary dataset model = (truePositives, trueNegatives, falsePositives, falseNegatives)
    where
        outputsTrain = map (\(input, actual) -> (actual, mlpLayer model (asTensor input))) dataset
        outputsTrain' = map (\(actual, prediction) -> (actual, if (asValue prediction :: Float) > 0.5 then 1 else 0)) outputsTrain
        truePositives = length $ filter (\(actual, prediction) -> prediction == 1 && actual == 1) outputsTrain'
        trueNegatives = length $ filter (\(actual, prediction) -> prediction == 0 && actual == 0) outputsTrain'
        falsePositives = length $ filter (\(actual, prediction) -> prediction == 1 && actual == 0) outputsTrain'
        falseNegatives = length $ filter (\(actual, prediction) -> prediction == 0 && actual == 1) outputsTrain'
        
-- f(tp, tn,total) = tp + tn / total
accuracy :: Int -> Int -> Int -> Float
accuracy tp tn total = fromIntegral (tp + tn) / fromIntegral total

-- f(tp,fp) = tp/(tp+fp)
precision :: Int -> Int -> Float
precision tp fp = fromIntegral tp / fromIntegral (tp + fp)

-- f(tp,fn) = tp/(tp+fn)
recall :: Int -> Int -> Float
recall tp fn = fromIntegral tp / fromIntegral (tp + fn)