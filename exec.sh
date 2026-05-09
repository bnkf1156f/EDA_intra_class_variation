#!/bin/bash

echo "============================================================"
echo "  INTRA-CLASS VARIATION EDA PIPELINE"
echo "============================================================"
echo ""
echo "  Which pipeline do you want to run?"
echo ""
echo "  1. Pre-Annotation                        (frame quality assessment - BEFORE annotation)"
echo "  2. Post-Annotation and Pre-Training      (intra-class analysis - AFTER annotation and BEFORE training)"
echo ""
echo "  0. Exit"
echo ""
echo "============================================================"
read -p "  Select [0-2]: " choice

echo ""

case "$choice" in
    1)
        echo "  Launching Pre-Annotation pipeline..."
        echo ""
        python "master_scripts/1. master_script_dinov2_PreAnn.py"
        ;;
    2)
        echo "  Launching Post-Annotation and Pre-Training pipeline..."
        echo ""
        python "master_scripts/1. master_script_dinov2_PostAnn_PreTrain.py"
        ;;
    0)
        echo "  Bye!"
        exit 0
        ;;
    *)
        echo "  Invalid choice. Please run again."
        ;;
esac

echo ""
read -p "Press Enter to continue..."
