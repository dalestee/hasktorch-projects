{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE RecordWildCards #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Unused LANGUAGE pragma" #-}
{-# HLINT ignore "Fuse mapM/map" #-}
{-# HLINT ignore "Missing NOINLINE pragma" #-}
{-# OPTIONS_GHC -Wno-name-shadowing #-}

module Main where

--hasktorch
import Control.Monad (when)
import Data.List()
import Torch.Tensor         (Tensor, asTensor)
import Torch.Functional     (mseLoss)
import Torch.NN             (sample)
import Torch.Optim          (GD(..), foldLoop)
import Torch.Device         (Device(..),DeviceType(..))
import Torch.Train          (update)
import Torch.Layer.MLP (MLPHypParams(..), ActName(..), mlpLayer, MLPParams)

-- other libraries for parsing
import System.IO.Unsafe (unsafePerformIO)
import Data.List.Split (splitOn)

--------------------------------------------------------------------------------
-- Data
--------------------------------------------------------------------------------

-- Survived,Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
parseDataTrain :: [([Float], Float)]
parseDataTrain = unsafePerformIO $ do
    content <- readFile "data/train-mod.csv"
    let lines' = tail $ lines content
    return $ map (\line -> let fields = splitOn "," line in
        ([read (fields !! 1), read (fields !! 2), read (fields !! 3), read (fields !! 4), read (fields !! 5), read (fields !! 6), read (fields !! 7)], read (fields !! 0))) lines'

-- Pclass,Sex,Age,SibSp,Parch,Fare,Embarked
parseDataTest :: [(Float, Float, Float, Float, Float, Float, Float)]
parseDataTest = unsafePerformIO $ do
    content <- readFile "data/test-mod.csv"
    let lines' = tail $ lines content
    return $ map (\line -> let fields = splitOn "," line in
        (read (head fields), read (fields !! 1), read (fields !! 2), read (fields !! 3), read (fields !! 4), read (fields !! 5), read (fields !! 6))) lines'

loss :: MLPParams -> ([Float], Float) -> Tensor
loss model (input, output) = let y = mlpLayer model (asTensor input) 
                             in mseLoss y (asTensor output)

main :: IO ()
main = do
    -- parse training data and print the first 5
    -- print $ Prelude.take 5 (parseDataTrain)
    -- print $ Prelude.take 5 (parseDataTest)
    -- return ()
    init <- sample hyperParams

    (trained, _, losses) <- foldLoop (init, opt, []) numIters $ \(model, optimizer, losses) i -> do
        let epochLoss = sum (map (loss model) trainingData)
        losses <- return $ losses ++ [epochLoss]
        when (i `mod` 1 == 0) $ do
          print i
          print epochLoss
        (newState, newOpt) <- update model optimizer epochLoss lr
        return (newState, newOpt, losses)
    return ()

    where
        numIters = 2000
        lr = 1e-4
        opt = GD
        trainingData = parseDataTrain
        device = Device CPU 0
        hyperParams = MLPHypParams device 7 [(21, Relu), (1, Id)]

--------------------------------------------------------------------------------
-- Test
--------------------------------------------------------------------------------

