#!/bin/bash

MODEL=$1

flatc -t _schema/schema_v3a.fbs -- $MODEL
