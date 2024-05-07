module Main (main) where

import System.Environment (getArgs)
import Titanic -- titanic
import LinearRegression -- linear

main :: IO ()
main = do
    args <- getArgs
    case args of
        ["titanic"] -> titanic
        ["linear"] -> linear
        _ -> putStrLn "Usage: stack run <titanic|linear>"