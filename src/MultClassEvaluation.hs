module MultClassEvaluation (accuracy, precision, recall, extractBinaryMultClassification) where

import Torch.Train        (loadParams)
import Torch.Layer.MLP    (mlpLayer, MLPHypParams(..))
import Torch.Tensor       (asValue, Tensor)
import Func (argmax)

extractBinaryMultClassification :: String -> [(Tensor, Tensor)] -> MLPHypParams -> IO ([Int], [Int], [Int])
extractBinaryMultClassification modelPath dataset hyperParams = do
    model <- loadParams hyperParams modelPath
    let outputsTrain = map (\(input, actual) -> (actual, mlpLayer model input)) dataset
    let outputsTrain' = map (\(actual, prediction) -> (actual, prediction)) outputsTrain
    -- if the index of the 1.0 inside the actualClasss is equal to the index of the biggest float inside the predictedClass then is a true positive
    let truePositives = map (\(actual, prediction) -> if argmax (asValue actual) == argmax (asValue prediction) then 1 else 0) outputsTrain'
    let falseNegatives = map (\(actual, prediction) -> if argmax (asValue actual) /= argmax (asValue prediction) then 1 else 0) outputsTrain'
    let falsePositives = map (\(actual, prediction) -> if argmax (asValue actual) /= argmax (asValue prediction) then 1 else 0) outputsTrain'
    return (truePositives, falseNegatives, falsePositives)

accuracy :: [Int] -> [Int] -> [Int] -> Float
accuracy tps tns fps = fromIntegral (sum tps + sum tns) / fromIntegral (sum tps + sum tns + sum fps)

precision :: [Int] -> [Int] -> Float
precision tps fps = fromIntegral (sum tps) / fromIntegral (sum tps + sum fps)

recall :: [Int] -> [Int] -> Float
recall tps fns = fromIntegral (sum tps) / fromIntegral (sum tps + sum fns)