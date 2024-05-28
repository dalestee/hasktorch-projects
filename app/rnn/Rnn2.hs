{-# OPTIONS_GHC -Wno-name-shadowing #-}
module Rnn2 (rnn2) where

import Torch.Layer.RNN      (RnnHypParams(..), RnnParams(..), rnnLayers)
import Torch.Tensor         (Tensor(..), asValue, asTensor, TensorLike (asTensor), shape)
import Torch.Functional     (Dim(..), mseLoss)
import Torch.Device         (Device(..), DeviceType (CPU))
import Torch.NN             (sample, flattenParameters)
import Torch.Optim          (mkAdam, foldLoop)
import Torch.Train          (loadParams, update, saveParams)
import Torch.Layer.MLP      (ActName(Relu, Tanh))
import Control.Monad        (when)
import Codec.Picture.Metadata (Value(Int))
import Torch (DType(Float))
import Torch.TensorFactories (zeros')


loss :: RnnParams -> Tensor -> (Tensor, Tensor) -> Tensor
loss model initialState (input, target) =
    let (output, _) = forward model initialState input
    in mseLoss output target

forward :: RnnParams -> Tensor -> Tensor -> (Tensor,Tensor) -- ^ an output of (<seqLen,D*oDim>,(<D*numLayers,oDim>,<D*numLayers,cDim>)) so (output, (final hidden state, final cell state))
forward model initialState input = output
    where
        output = rnnLayers model Relu Nothing initialState input

-- Function to generate a single data pair based on an index
generateDataPair :: Int -> (Tensor, Tensor)
generateDataPair i = 
    let input = [[fromIntegral i + 2], [fromIntegral i + 1], [fromIntegral i]]
        output = [[fromIntegral i-1]]
    in (asTensor (input :: [[Float]]), asTensor (output :: [[Float]]))

-- Function to create the training data up to a specified size
createTrainingData :: Int -> [(Tensor, Tensor)]
createTrainingData n = [generateDataPair i | i <- [0..(n-1)]]

rnn2 :: IO ()
rnn2 = do
    putStrLn "Rnn"

    -- init <- loadParams hyperParams modelPath -- comment if you want to sample a model
    init <- sample hyperParams -- comment if you want to load a model

    print initialState

    -- ^ an input tensor <seqLen,iDim> where seqLen is the sequence length and iDim is the input dimension
    let inputData = asTensor ([[5], [4], [3]] :: [[Float]])

    print $ shape initialState
    print $ shape inputData

    let (output1,_) = forward init initialState inputData
    putStrLn "Output before training"
    print output1
    let trainingData = createTrainingData 100

    print $ take 1 trainingData
    
    let opt = mkAdam itr beta1 beta2 (flattenParameters init)

    putStrLn "Start training"
    (trained, _, losses) <- foldLoop (init, opt, []) numEpochs $ \(model, optimizer, losses) i -> do
        let epochLoss = sum (map (loss model initialState) trainingData)
        when (i `mod` 1 == 0) $ do
            print ("Epoch: " ++ show i ++ " | Loss: " ++ show (asValue epochLoss :: Float))
        -- when (i `mod` 50 == 0) $ do
        --     saveParams model (modelPath ++ "-" ++ show i ++ ".pt")
        (newState, newOpt) <- update model optimizer epochLoss lr
        return (newState, newOpt, losses :: [Float]) -- without the losses curve
        -- return (newState, newOpt, losses ++ [asValue epochLoss :: Float]) -- with the losses curve

    print "Training done"

    let (output2,_) = forward trained initialState inputData
    putStrLn "Output after training"
    print output2

    print "Done!"

    where
        numEpochs = 300 :: Int

        modelPath :: String
        modelPath =  "app/rnn/models/sample_model.pt"

        numLayers = 1
        hDim = 2
        iDim = 1

        hyperParams :: RnnHypParams
        hyperParams = RnnHypParams {
            dev = Device CPU 0,
            bidirectional = False, -- ^ True if BiLSTM, False otherwise
            inputSize = iDim, -- ^ The number of expected features in the input x
            hiddenSize = hDim, -- ^ The number of features in the hidden state h
            numLayers = numLayers, -- ^ Number of recurrent layers
            hasBias = True -- ^ If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        }

        -- betas are decaying factors Float, m's are the first and second moments [Tensor] and iter is the iteration number Int
        itr = 0 :: Int
        beta1 = 0.9 :: Float
        beta2 = 0.999 :: Float
        lr = 1e-1 :: Tensor
        dim = Dim 0 :: Dim

        initialState = zeros' [numLayers, hDim] :: Tensor