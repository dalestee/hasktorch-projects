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
import Torch.NN             (sample, flattenParameters)
import Torch.Optim          (Adam(..), foldLoop, mkAdam, GD(..))
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update, saveParams, loadParams)
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer, MLPParams)

-- other libraries for parsing
import System.IO.Unsafe (unsafePerformIO)
import Data.List.Split (splitOn)
import ML.Exp.Chart (drawLearningCurve) --nlp-tools

--------------------------------------------------------------------------------
-- Data
--------------------------------------------------------------------------------

-- Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
parseData :: IO ([([Float], Float)], [([Float], Float)])
parseData = do
    content <- readFile "app/titanic-mlp/data/train-mod.csv"
    let lines' = tail $ lines content
    let dataset = map (\line -> let fields = splitOn "," line in ([read (fields !! 1), read (fields !! 2), read (fields !! 3), read (fields !! 4), read (fields !! 5), read (fields !! 6), read (fields !! 7)], read (fields !! 0))) lines'
    let trainSize = round (0.9 * fromIntegral (length dataset)) :: Int
    let (trainData, valData) = splitAt trainSize dataset
    return (trainData, valData)

-- Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
parseDataTest :: [([Float], Int)]
parseDataTest = unsafePerformIO $ do
    content <- readFile "app/titanic-mlp/data/test-mod.csv"
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
    -- init <- sample hyperParams
    init <- loadParams hyperParams "app/titanic-mlp/models/model-titanic-160.46135.pt"
    let opt = mkAdam itr beta1 beta2 (flattenParameters init)
    -- let opt = GD
    (trained, _, losses) <- foldLoop (init, opt, []) numIters $ \(model, optimizer, losses) i -> do
        let epochLoss = sum (map (loss model) trainingData)
        when (i `mod` 1 == 0) $ do
            print i
            print epochLoss
        (newState, newOpt) <- update model optimizer epochLoss lr
        -- return (newState, newOpt, losses :: [Float]) -- without the losses curve
        return (newState, newOpt, losses ++ [asValue epochLoss :: Float]) -- with the losses curve

    let maybeLastLoss = if null losses then Nothing else Just (last losses)
    case maybeLastLoss of
        Nothing -> return ()
        Just lastLoss -> do
            let filename = "app/titanic-mlp/curves/graph-titanic-mse" ++ show lastLoss ++ ".png"
            drawLearningCurve filename "Learning Curve" [("", losses)]

    let modelName = "app/titanic-mlp/models/model-titanic-" ++
                    (if null losses then "noLosses" else show (last losses)) ++ ".pt"
    saveParams trained modelName

    model <- loadParams hyperParams modelName

    let outputsTrain = map (\(input, passengerId) -> (passengerId, mlpLayer model (asTensor input))) validationData
    let outputsTrain' = map (\(passengerId, output) -> (passengerId, if (asValue output :: Float) > 0.5 then 1 else 0)) outputsTrain
    let successRate = fromIntegral (length (filter (uncurry (==)) outputsTrain')) / fromIntegral (length outputsTrain')
    putStrLn $ "Success rate on validation data: " ++ show successRate

    let testData = parseDataTest
    -- test 1 output
    let outputs = map (\(input, passengerId) -> (passengerId, mlpLayer model (asTensor input))) testData
    let outputs' = map (\(passengerId, output) -> (passengerId, if (asValue output :: Float) > 0.5 then 1 else 0)) outputs
    writeFile "app/titanic-mlp/data/submission.csv" $ arrayToCSV outputs'
    return ()

    where
        numIters = 0
        device = Device CPU 0
        hyperParams = MLPHypParams device 7 [(21, Relu), (1, Id)]
        -- betas are decaying factors Float, m's are the first and second moments [Tensor] and iter is the iteration number Int
        itr = 0
        beta1 = 0.9
        beta2 = 0.999
        lr = 1e-2
        trainingData = unsafePerformIO $ fst <$> parseData
        validationData = unsafePerformIO $ snd <$> parseData

--------------------------------------------------------------------------------
-- Validation
--------------------------------------------------------------------------------

