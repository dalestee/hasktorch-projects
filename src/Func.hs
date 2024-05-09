module Func (argmax) where

argmax :: [Float] -> Int
argmax xs = snd $ maximum $ zip xs [0..]