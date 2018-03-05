all:
	@echo "Usage: make [train | tensorboard | monitor]"

train:
	python3 -u train.py

tensorboard:
	@echo "Starting tensorboard in background... (use \"tmux a\" to attach)"
	@tmux new -d 'tensorboard --logdir=logs'
	@tmux ls

monitor:
	@glances
