{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Unused LANGUAGE pragma" #-}
{-# HLINT ignore "Fuse mapM/map" #-}
{-# HLINT ignore "Missing NOINLINE pragma" #-}
{-# OPTIONS_GHC -Wno-name-shadowing #-}

module Titanic (titanic) where

--hasktorch
import Control.Monad (when)
import Data.List()
import Torch.Tensor         (Tensor, asTensor, asValue)
import Torch.Functional     (mseLoss)
import Torch.NN             (sample)
import Torch.Optim          (GD(..), foldLoop)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer, MLPParams)

-- other libraries for parsing
import System.IO.Unsafe (unsafePerformIO)
import Data.List.Split (splitOn)
import ML.Exp.Chart (drawLearningCurve) --nlp-tools
import Torch (asValue, DType (Float))

--------------------------------------------------------------------------------
-- Data
--------------------------------------------------------------------------------

-- Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
parseDataTrain :: [([Float], Float)]
parseDataTrain = unsafePerformIO $ do
    content <- readFile "src/titanic-mlp/data/train-mod.csv"
    let lines' = tail $ lines content
    return $ map (\line -> let fields = splitOn "," line in
        ([read (fields !! 1), read (fields !! 2), read (fields !! 3), read (fields !! 4), read (fields !! 5), read (fields !! 6), read (fields !! 7)], read (fields !! 0))) lines'

-- Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
parseDataTest :: [([Float], Int)]
parseDataTest = unsafePerformIO $ do
    content <- readFile "src/titanic-mlp/data/test-mod.csv"
    let lines' = tail $ lines content
    return $ map (\line -> let fields = splitOn "," line in
        ([read (fields !! 1), read (fields !! 2), read (fields !! 3), read (fields !! 4), read (fields !! 5), read (fields !! 6), read (fields !! 7)], read (fields !! 0))) lines'

loss :: MLPParams -> ([Float], Float) -> Tensor
loss model (input, output) = let y = mlpLayer model (asTensor input)
                             in mseLoss y (asTensor output)

arrayToCSV :: [(Int, Int)] -> String
arrayToCSV = unlines . map (\(a, b) -> show a ++ "," ++ show b)

titanic :: IO ()
titanic = do
    init <- sample hyperParams
    -- init <- loadParams hyperParams "src/titanic-mlp/model-titanic.pt"

    (trained, _, losses) <- foldLoop (init, opt, []) numIters $ \(model, optimizer, losses) i -> do
        let epochLoss = sum (map (loss model) trainingData)
        when (i `mod` 1 == 0) $ do
            print i
            print epochLoss
        (newState, newOpt) <- update model optimizer epochLoss lr
        return (newState, newOpt, losses ++ [asValue epochLoss :: Float])

    let filename = "src/titanic-mlp/curves/graph-titanic-mse" ++ show (last losses) ++ ".png"
    let modelName = "src/titanic-mlp/models/model-titanic-" ++ show (last losses) ++ ".pt"
    drawLearningCurve filename "Learning Curve" [("", losses)]
    saveParams trained modelName

    model <- loadParams hyperParams modelName

    let trainDataTest = parseDataTrain
    let outputsTrain = map (\(input, passengerId) -> (passengerId, mlpLayer model (asTensor input))) trainDataTest
    let outputsTrain' = map (\(passengerId, output) -> (passengerId, if (asValue output :: Float) > 0.5 then 1 else 0)) outputsTrain
    let successRate = fromIntegral (length (filter (uncurry (==)) outputsTrain')) / fromIntegral (length outputsTrain')
    putStrLn $ "Success rate on training data: " ++ show successRate

    let testData = parseDataTest
    -- test 1 output
    let outputs = map (\(input, passengerId) -> (passengerId, mlpLayer model (asTensor input))) testData
    let outputs' = map (\(passengerId, output) -> (passengerId, if (asValue output :: Float) > 0.5 then 1 else 0)) outputs
    writeFile "src/titanic-mlp/data/submission.csv" $ arrayToCSV outputs'
    return ()

    where
        numIters = 60
        lr = 0.0001
        opt = GD
        trainingData = parseDataTrain
        device = Device CPU 0
        hyperParams = MLPHypParams device 7 [(21, Relu), (1, Id)]

--------------------------------------------------------------------------------
-- Validation
--------------------------------------------------------------------------------

