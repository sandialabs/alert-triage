import ConfigParser
import argparse
import logging

from alert_triage.batch_pipeline.batch_pipeline_scot import (
    SCOTTrainPipeline, SCOTTestPipeline, SCOTListenPipeline)

def main():
    parser = argparse.ArgumentParser(description="Runs the batch pipeline.\n" +
                "This can be run in either training mode, testing mode, or.\n" +
                "testing mode.  Training mode collects all promoted events\n" +
                "and the past three months of closed events (unless limited\n" +
                "with the options.  Testing collects the past x days of\n" +
                "open alerts and applies the trained classifier to them\n" +
                "Listen actively listens for new alerts and applies the\n" +
                "classifier in realtime\n")
    parser.add_argument('--limit',
                        type=int,
                        default=0,
                        help="Limit the number of alerts (0 for all) for " +
                          "training and testing pipelines.")
    parser.add_argument('--config',
                        type=str,
                        default="config.ini",
                        help="Where the config file is located.")
    parser.add_argument('--classifier',
                        type=str,
                        help="Where to store the model after training.")
    parser.add_argument('--debug',
                        action='store_true',
                        help="Enables logging")
    parser.add_argument('--days',
                        default=7,
                        type=int,
                        help="The number of days-worth of alerts to" +
                            " grab and either train from or classify.")
    parser.add_argument('--excludeCachedCorrelation',
                        action='store_true',
                        help="Excludes cached correlation feature extraction")
    parser.add_argument('--excludeExtractedEntities',
                        action='store_true',
                        help="Excludes entity extraction")
    parser.add_argument('--excludeRaw',
                        action='store_true',
                        help="Excludes raw feature extraction")
    parser.add_argument('--excludeLDA',
                        action='store_true',
                        help="Excludes lda feature extraction")
    parser.add_argument('--train',
                        action='store_true',
                        help="Run the training pipeline")
    parser.add_argument('--test',
                        action='store_true',
                        help="Run the testing pipeline")
    parser.add_argument('--listen',
                        action='store_true',
                        help="Run pipeline in listening mode")
    args = parser.parse_args()
    limit      = args.limit
    configFile = args.config
    debug      = args.debug        
    days             = args.days
    pickleFile       = args.classifier
    includeCachedCorrelation = not args.excludeCachedCorrelation
    includeExtractedEntities = not args.excludeExtractedEntities
    includeRaw = not args.excludeRaw
    includeLDA = not args.excludeLDA
    train = args.train
    test  = args.test
    listen = args.listen

    if debug:
        logging.basicConfig(level=logging.DEBUG,
            format="%(levelname)s:%(filename)s.%(funcName)s:%(message)s");

    if train:
        logging.info("Running the training pipeline")


        pipeline = SCOTTrainPipeline(configFile = configFile,
                         days = days,
                         limit = limit,
                         pickleFile = pickleFile,
                         includeCachedCorrelation = includeCachedCorrelation,
                         includeExtractedEntities = includeExtractedEntities,
                         includeLDA = includeLDA,
                         includeRaw = includeRaw)

        pipeline.run()
    elif test:
        logging.info("Running the testing pipeline")
        pipeline = SCOTTestPipeline(configFile = configFile,
                                      days = days,
                                      limit = limit,
                                      pickleFile = pickleFile)
        pipeline.run()

    elif listen:
        logging.info("Running in listening mode")
        pipeline = SCOTListenPipeline(configFile = configFile,
                      pickleFile = pickleFile)
        pipeline.run()
    else:
        print "Neither training, testing, nor listen selected"






main()
