module MultClassEvaluation (accuracy, confusionMatrix, confusionMatrix', f1Macro, f1Micro, f1Weighted, fp, tp, fn) where

import Torch.Train        (loadParams)
import Torch.Layer.MLP    (mlpLayer, MLPHypParams(..), MLPParams)
import Torch.Tensor       (asValue, Tensor)
import Func (argmax)

tp :: [[Int]] -> Int -> IO Int
tp confMat classIndex = do
    return (confMat !! classIndex !! classIndex)

fp :: [[Int]] -> Int -> IO Int
fp confMat classIndex = do
    let numberOfClasses = length confMat
    let row = confMat !! classIndex
    return $ sum (map (row !!) [0..numberOfClasses-1]) - row !! classIndex

fn :: [[Int]] -> Int -> IO Int
fn confMat classIndex = do
    let col = map (!! classIndex) confMat
    return $ sum col - col !! classIndex


accuracy :: [[Int]] -> IO Float
accuracy confMat = do
    let numberOfClasses = length confMat
    tps <- mapM (tp confMat) [0..numberOfClasses-1]
    let numberOfElements = sum (map sum confMat)
    let accuracyValue = fromIntegral (sum tps) / fromIntegral numberOfElements
    return accuracyValue

confusionMatrix :: String -> [(Tensor, Tensor)] -> MLPHypParams -> IO [[Int]]
confusionMatrix  modelPath dataset hyperParams = do
    model <- loadParams hyperParams modelPath
    let outputsTrain = map (\(input, actual) -> (actual, mlpLayer model input)) dataset
    let outputsTrain' = map (\(actual, prediction) -> (actual, prediction)) outputsTrain
    -- foreach (actual, prediction) in outputsTrain' actual is i and prediction is j
    -- create a matrix of size 10x10
    -- and add one to matrix[i][j]
    let matrix = [[0 | _ <- [(0 :: Integer)..9]] | _ <- [(0 :: Integer)..9]]
    let matrix' = foldl (\acc (actual, prediction) -> do
            let i = argmax (asValue actual)
            let j = argmax (asValue prediction)
            let row = acc !! i
            let row' = take j row ++ [row !! j + 1] ++ drop (j + 1) row
            let acc' = take i acc ++ [row'] ++ drop (i + 1) acc
            acc') matrix outputsTrain'
    return matrix'

confusionMatrix' :: MLPParams -> [(Tensor, Tensor)] -> IO [[Int]]
confusionMatrix'  model dataset = do
    let outputsTrain = map (\(input, actual) -> (actual, mlpLayer model input)) dataset
    let outputsTrain' = map (\(actual, prediction) -> (actual, prediction)) outputsTrain
    -- foreach (actual, prediction) in outputsTrain' actual is i and prediction is j
    -- create a matrix of size 10x10
    -- and add one to matrix[i][j]
    let matrix = [[0 | _ <- [(0 :: Integer)..9]] | _ <- [(0 :: Integer)..9]]
    let matrix' = foldl (\acc (actual, prediction) -> do
            let i = argmax (asValue actual)
            let j = argmax (asValue prediction)
            let row = acc !! i
            let row' = take j row ++ [row !! j + 1] ++ drop (j + 1) row
            let acc' = take i acc ++ [row'] ++ drop (i + 1) acc
            acc') matrix outputsTrain'
    return matrix'

f1 :: [[Int]] -> Int -> IO Float
f1 confMat classIndex = do
    -- returns f1 score of one class
    tp' <- tp confMat classIndex
    fp' <- fp confMat classIndex
    fn' <- fn confMat classIndex
    let precisionValue = fromIntegral tp' / (fromIntegral tp' + fromIntegral fp')
    let recallValue = fromIntegral tp' / (fromIntegral tp' + fromIntegral fn')
    let f1Value = 2 * precisionValue * recallValue / (precisionValue + recallValue)
    return f1Value

f1Macro :: [[Int]] -> IO Float
f1Macro confMat = do
    let numberOfClasses = length confMat
    f1s <- mapM (f1 confMat) [0..numberOfClasses-1]
    let f1MacroValue = sum f1s / fromIntegral (length f1s)
    return f1MacroValue


f1Micro :: [[Int]] -> IO Float
f1Micro confMat = do
    let numberOfClasses = length confMat
    sumTps <- mapM (tp confMat) [0..numberOfClasses-1]
    sumFps <- mapM (fp confMat) [0..numberOfClasses-1]
    sumFns <- mapM (fn confMat) [0..numberOfClasses-1]
    -- TPs / TPs+(0.5(FPs+FNs))
    let micro = fromIntegral (sum sumTps) / (fromIntegral (sum sumTps) + 0.5 * (fromIntegral (sum sumFps) + fromIntegral (sum sumFns)))
    return micro

f1Weighted :: [[Int]] -> IO Float
f1Weighted confMat = do
    let numberOfClasses = length confMat
    f1s <- mapM (f1 confMat) [0..numberOfClasses-1]
    let weights = map (\classIndex -> fromIntegral (length (confMat !! classIndex))) [0..numberOfClasses-1]
    let f1WeightedValue = sum (zipWith (*) f1s weights) / sum weights
    return f1WeightedValue
