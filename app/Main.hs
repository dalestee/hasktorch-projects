module Main (main) where

import System.Environment (getArgs)
import Titanic -- titanic
import LinearRegression -- linear
import Cifar -- cifar
import Temperature -- temperature

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["titanic-mlp"] -> titanic
        ["linear"] -> linear
        ["cifar"] -> cifar
        ["temperature"] -> temperature
        _ -> putStrLn "Usage: stack run <titanic|linear|cifar|temperature>"