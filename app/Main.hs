module Main (main) where

import System.Environment (getArgs)
import Titanic -- titanic
import LinearRegression -- linear
import Cifar -- cifar

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["titanic"] -> titanic
        ["linear"] -> linear
        ["cifar"] -> cifar
        _ -> putStrLn "Usage: stack run <titanic|linear|cifar>"