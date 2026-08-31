# ----------------
#  Arguments
# ----------------
DATA_DIR := $(or $(DATA_DIR), ../data)

# ----------------
# Definitions
# ----------------
datasets_list := $(or $(DS_LIST), CIFAR100 ImageNet)
models := $(or $(MODELS), VGG MobileNet ResNet ConvNeXt)
reductions := $(or $(REDUCTIONS), kernel toeplitz avgpooling)
analyses := $(or $(ANALYSES), MACS DMD)

# ----------------
# Parsing names
# ----------------
datasets := $(foreach ds, $(datasets_list), $(foreach m, $(models), $(DATA_DIR)/$(ds)/datasets/dss.$(ds)-test.APGD-t-$(m)))

corevectors := $(foreach ds, $(datasets_list), $(foreach p, $(foreach m, $(models), $(DATA_DIR)/$(ds)/$(m)/corevectors), $(foreach r, $(reductions), $(p)/$(r))))

peepholes := $(foreach ds, $(datasets_list), $(foreach pp, $(foreach p, $(foreach m, $(models), $(DATA_DIR)/$(ds)/$(m)/peepholes), $(foreach r, $(reductions), $(p)/$(r))), $(foreach a, $(analyses), $(pp)/$(a))))

tunings := $(foreach ds, $(datasets_list), $(foreach pp, $(foreach p, $(foreach m, $(models), $(DATA_DIR)/$(ds)/$(m)/peepholes), $(foreach r, $(reductions), $(p)/$(r))), $(foreach a, $(analyses), $(pp)/$(a)/hyperparams.pickle)))

# ----------------
#  Rules
# ----------------
.PHONY: $(peepholes)
#
# ----------------
# Accessible Rules
# ----------------
xp_datasets: $(datasets)

xp_corevectors: $(corevectors)

xp_peepholes: $(peepholes)

xp_tuning: $(tunings)

# ----------------
# Actual Rules
# ----------------

#  Datasets
define ds_template =
$(1):
	python datasets/xp_datasets.py -ds $(2) -m $(3) -d $(DATA_DIR)
endef
$(foreach ds, $(datasets_list), $(foreach m, $(models), $(eval $(call ds_template, $(DATA_DIR)/$(ds)/datasets/dss.$(ds)-test.APGD-t-$(m), $(ds), $(m)))))

# Corevectors
define cvs_template =
$(1):
	python corevectors/xp_corevectors.py -m $(2) -r $(3) -ds $(4) -d $(DATA_DIR)
endef
$(foreach ds, $(datasets_list), $(foreach r, $(reductions), $(foreach m, $(models), $(eval $(call cvs_template, $(DATA_DIR)/$(ds)/$(m)/corevectors/$(r), $(m), $(r), $(ds))))))

# Peepholes
define phs_template =
$(1):
	python aucs/xp_peepholes.py -m $(2) -r $(3) -a $(4) -ds $(5) -d $(DATA_DIR)
	python aucs/xp_others.py -m $(2) -ds $(5) -d $(DATA_DIR)
endef
$(foreach ds, $(datasets_list), $(foreach a, $(analyses), $(foreach r, $(reductions), $(foreach m, $(models), $(eval $(call phs_template, $(DATA_DIR)/$(ds)/$(m)/peepholes/$(r)/$(a), $(m), $(r), $(a), $(ds)))))))

# Tunings
define tune_template =
$(1):
	python tuning/xp_tuning.py -m $(2) -r $(3) -a $(4) -ds $(5) -d $(DATA_DIR)
endef
$(foreach ds, $(datasets_list), $(foreach a, $(analyses), $(foreach r, $(reductions), $(foreach m, $(models), $(eval $(call tune_template, $(DATA_DIR)/$(ds)/$(m)/peepholes/$(r)/$(a)/hyperparams.pickle, $(m), $(r), $(a), $(ds)))))))
