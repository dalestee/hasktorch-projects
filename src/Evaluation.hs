module Evaluation (accuracy, precision, recall, extractBinary) where

import Torch.Train        (loadParams)
import Torch.Layer.MLP    (mlpLayer, MLPHypParams(..))
import Torch.Tensor       (asTensor, asValue)

-- returns (tp, tn, fp, fn)
extractBinary :: String -> [([Float], Float)] -> MLPHypParams -> IO (Int, Int, Int, Int)
extractBinary modelPath dataset hyperParams = do
    model <- loadParams hyperParams modelPath
    let outputsTrain = map (\(input, actual) -> (actual, mlpLayer model (asTensor input))) dataset
    let outputsTrain' = map (\(actual, prediction) -> (actual, if (asValue prediction :: Float) > 0.5 then 1 else 0)) outputsTrain
    let truePositives = length $ filter (\(actual, prediction) -> prediction == 1 && actual == 1) outputsTrain'
    let trueNegatives = length $ filter (\(actual, prediction) -> prediction == 0 && actual == 0) outputsTrain'
    let falsePositives = length $ filter (\(actual, prediction) -> prediction == 1 && actual == 0) outputsTrain'
    let falseNegatives = length $ filter (\(actual, prediction) -> prediction == 0 && actual == 1) outputsTrain'
    return (truePositives, trueNegatives, falsePositives, falseNegatives)

-- f(tp, tn,total) = tp + tn / total
accuracy :: Int -> Int -> Int -> Float
accuracy tp tn total = fromIntegral (tp + tn) / fromIntegral total

-- f(tp,fp) = tp/(tp+fp)
precision :: Int -> Int -> Float
precision tp fp = fromIntegral tp / fromIntegral (tp + fp)

-- f(tp,fn) = tp/(tp+fn)
recall :: Int -> Int -> Float
recall tp fn = fromIntegral tp / fromIntegral (tp + fn)