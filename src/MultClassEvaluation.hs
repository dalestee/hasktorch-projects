module MultClassEvaluation (accuracy, confusionMatrix, f1Macro, f1Micro, f1Weighted, fp, tp, fn, avarage, variance) where

import Torch.Train        (loadParams)
import Torch.Layer.MLP    (mlpLayer, MLPHypParams(..), MLPParams)
import Torch.Tensor       (asValue, Tensor)
import Func (argmax)

tp :: [[Int]] -> Int -> Int
tp confMat classIndex = confMat !! classIndex !! classIndex

fp :: [[Int]] -> Int -> Int
fp confMat classIndex = sum (map (row !!) [0..numberOfClasses-1]) - row !! classIndex
    where
        numberOfClasses = length confMat
        row = confMat !! classIndex

fn :: [[Int]] -> Int -> Int
fn confMat classIndex = sum col - col !! classIndex
    where
        col = map (!! classIndex) confMat


accuracy :: [[Int]] -> Float
accuracy confMat = fromIntegral (sum tps) / fromIntegral numberOfElements
    where
        numberOfClasses = length confMat
        tps = map (tp confMat) [0..numberOfClasses-1]
        numberOfElements = sum (map sum confMat)

confusionMatrix :: MLPParams -> [(Tensor, Tensor)] -> [[Int]]
confusionMatrix model dataset = matrix'
  where
    outputsTrain = map (\(input, actual) -> (actual, mlpLayer model input)) dataset
    outputsTrain' = map (\(actual, prediction) -> (actual, prediction)) outputsTrain
    matrix = [[0 | _ <- [(0 :: Integer)..9]] | _ <- [(0 :: Integer)..9]]
    matrix' = foldl updateMatrix matrix outputsTrain'
    updateMatrix acc (actual, prediction) = 
        let i = argmax (asValue actual)
            j = argmax (asValue prediction)
            row = acc !! i
            row' = take j row ++ [row !! j + 1] ++ drop (j + 1) row
            acc' = take i acc ++ [row'] ++ drop (i + 1) acc
        in acc'

-- returns f1 score of one class
f1 :: [[Int]] -> Int -> Float
f1 confMat classIndex = 2 * precisionValue * recallValue / (precisionValue + recallValue)
    where 
        tp' = tp confMat classIndex
        fp' = fp confMat classIndex
        fn' = fn confMat classIndex
        precisionValue = fromIntegral tp' / (fromIntegral tp' + fromIntegral fp')
        recallValue = fromIntegral tp' / (fromIntegral tp' + fromIntegral fn')

f1Macro :: [[Int]] -> Float
f1Macro confMat = sum f1s / fromIntegral (length f1s)
    where
        numberOfClasses = length confMat
        f1s = map (f1 confMat) [0..numberOfClasses-1]

-- TPs / TPs+(0.5(FPs+FNs))
f1Micro :: [[Int]] -> Float
f1Micro confMat = fromIntegral (sum sumTps) / (fromIntegral (sum sumTps) + 0.5 * (fromIntegral (sum sumFps) + fromIntegral (sum sumFns)))
    where 
        numberOfClasses = length confMat
        sumTps = map (tp confMat) [0..numberOfClasses-1]
        sumFps = map (fp confMat) [0..numberOfClasses-1]
        sumFns = map (fn confMat) [0..numberOfClasses-1]


f1Weighted :: [[Int]] -> Float
f1Weighted confMat = sum (zipWith (*) f1s weights) / sum weights
    where
        numberOfClasses = length confMat
        f1s = map (f1 confMat) [0..numberOfClasses-1]
        weights = map (\classIndex -> fromIntegral (length (confMat !! classIndex))) [0..numberOfClasses-1]


avarage :: [Float] -> Float
avarage xs = sum xs / fromIntegral (length xs)

variance :: [Float] -> Float
variance xs = avarage $ map (\x -> (x - avarage xs) ^ 2) xs