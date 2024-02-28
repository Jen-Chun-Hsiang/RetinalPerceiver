from datasets.simulated_target_rf import CellClassLevel, ExperimentalLevel

# Create cells and cell classes
cell_class1_layout1 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]), class_level_id=1,
                                     sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                     sf_weight_surround=0.5, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class1_layout2 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]), class_level_id=1,
                                     sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                     sf_weight_surround=0.5, num_cells=8, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class1_layout3 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]), class_level_id=1,
                                     sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                     sf_weight_surround=0.5, num_cells=7, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class1_layout4 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]), class_level_id=1,
                                     sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                     sf_weight_surround=0.5, num_cells=9, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class1_layout5 = CellClassLevel(sf_cov_center=np.array([[0.12, 0.05], [0.04, 0.03]]), class_level_id=1,
                                     sf_cov_surround=np.array([[0.24, 0.05], [0.04, 0.06]]),
                                     sf_weight_surround=0.5, num_cells=5, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

cell_class2_layout1 = CellClassLevel(sf_cov_center=np.array([[0.08, 0.03], [0.06, 0.16]]), class_level_id=2,
                                     sf_cov_surround=np.array([[0.16, 0.03], [0.06, 0.32]]),
                                     sf_weight_surround=0.3, num_cells=8, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class2_layout2 = CellClassLevel(sf_cov_center=np.array([[0.08, 0.03], [0.06, 0.16]]), class_level_id=2,
                                     sf_cov_surround=np.array([[0.16, 0.03], [0.06, 0.32]]),
                                     sf_weight_surround=0.3, num_cells=11, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class2_layout3 = CellClassLevel(sf_cov_center=np.array([[0.08, 0.03], [0.06, 0.16]]), class_level_id=2,
                                     sf_cov_surround=np.array([[0.16, 0.03], [0.06, 0.32]]),
                                     sf_weight_surround=0.3, num_cells=13, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class2_layout4 = CellClassLevel(sf_cov_center=np.array([[0.08, 0.03], [0.06, 0.16]]), class_level_id=2,
                                     sf_cov_surround=np.array([[0.16, 0.03], [0.06, 0.32]]),
                                     sf_weight_surround=0.3, num_cells=12, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class2_layout5 = CellClassLevel(sf_cov_center=np.array([[0.08, 0.03], [0.06, 0.16]]), class_level_id=2,
                                     sf_cov_surround=np.array([[0.16, 0.03], [0.06, 0.32]]),
                                     sf_weight_surround=0.3, num_cells=10, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

cell_class3_layout1 = CellClassLevel(sf_cov_center=np.array([[0.1, 0.01], [0.01, 0.1]]), class_level_id=3,
                                     sf_cov_surround=np.array([[0.2, 0.01], [0.01, 0.2]]),
                                     sf_weight_surround=0.5, num_cells=16, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class3_layout2 = CellClassLevel(sf_cov_center=np.array([[0.1, 0.01], [0.01, 0.1]]), class_level_id=3,
                                     sf_cov_surround=np.array([[0.2, 0.01], [0.01, 0.2]]),
                                     sf_weight_surround=0.5, num_cells=14, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class3_layout3 = CellClassLevel(sf_cov_center=np.array([[0.1, 0.01], [0.01, 0.1]]), class_level_id=3,
                                     sf_cov_surround=np.array([[0.2, 0.01], [0.01, 0.2]]),
                                     sf_weight_surround=0.5, num_cells=14, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class3_layout4 = CellClassLevel(sf_cov_center=np.array([[0.1, 0.01], [0.01, 0.1]]), class_level_id=3,
                                     sf_cov_surround=np.array([[0.2, 0.01], [0.01, 0.2]]),
                                     sf_weight_surround=0.5, num_cells=15, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class3_layout5 = CellClassLevel(sf_cov_center=np.array([[0.1, 0.01], [0.01, 0.1]]), class_level_id=3,
                                     sf_cov_surround=np.array([[0.2, 0.01], [0.01, 0.2]]),
                                     sf_weight_surround=0.5, num_cells=13, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

cell_class4_layout1 = CellClassLevel(sf_cov_center=np.array([[0.12, -0.05], [-0.04, 0.03]]), class_level_id=4,
                                     sf_cov_surround=np.array([[0.24, -0.05], [-0.04, 0.06]]),
                                     sf_weight_surround=0.5, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class4_layout2 = CellClassLevel(sf_cov_center=np.array([[0.12, -0.05], [-0.04, 0.03]]), class_level_id=4,
                                     sf_cov_surround=np.array([[0.24, -0.05], [-0.04, 0.06]]),
                                     sf_weight_surround=0.5, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class4_layout3 = CellClassLevel(sf_cov_center=np.array([[0.12, -0.05], [-0.04, 0.03]]), class_level_id=4,
                                     sf_cov_surround=np.array([[0.24, -0.05], [-0.04, 0.06]]),
                                     sf_weight_surround=0.5, num_cells=7, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class4_layout4 = CellClassLevel(sf_cov_center=np.array([[0.12, -0.05], [-0.04, 0.03]]), class_level_id=4,
                                     sf_cov_surround=np.array([[0.24, -0.05], [-0.04, 0.06]]),
                                     sf_weight_surround=0.5, num_cells=5, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class4_layout5 = CellClassLevel(sf_cov_center=np.array([[0.12, -0.05], [-0.04, 0.03]]), class_level_id=4,
                                     sf_cov_surround=np.array([[0.24, -0.05], [-0.04, 0.06]]),
                                     sf_weight_surround=0.5, num_cells=7, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

cell_class5_layout1 = CellClassLevel(sf_cov_center=np.array([[0.12, 0], [-0.05, 0.03]]), class_level_id=5,
                                     sf_cov_surround=np.array([[0.20, 0], [-0.05, 0.05]]),
                                     sf_weight_surround=0.4, num_cells=9, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class5_layout2 = CellClassLevel(sf_cov_center=np.array([[0.12, 0], [-0.05, 0.03]]), class_level_id=5,
                                     sf_cov_surround=np.array([[0.20, 0], [-0.05, 0.05]]),
                                     sf_weight_surround=0.4, num_cells=10, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class5_layout3 = CellClassLevel(sf_cov_center=np.array([[0.12, 0], [-0.05, 0.03]]), class_level_id=5,
                                     sf_cov_surround=np.array([[0.20, 0], [-0.05, 0.05]]),
                                     sf_weight_surround=0.4, num_cells=9, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class5_layout4 = CellClassLevel(sf_cov_center=np.array([[0.12, 0], [-0.05, 0.03]]), class_level_id=5,
                                     sf_cov_surround=np.array([[0.20, 0], [-0.05, 0.05]]),
                                     sf_weight_surround=0.4, num_cells=10, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class5_layout5 = CellClassLevel(sf_cov_center=np.array([[0.12, 0], [-0.05, 0.03]]), class_level_id=5,
                                     sf_cov_surround=np.array([[0.20, 0], [-0.05, 0.05]]),
                                     sf_weight_surround=0.4, num_cells=11, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))

cell_class6_layout1 = CellClassLevel(sf_cov_center=np.array([[0.03, 0.05], [-0.05, 0.12]]), class_level_id=6,
                                     sf_cov_surround=np.array([[0.07, 0.05], [-0.05, 0.30]]),
                                     sf_weight_surround=0.7, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class6_layout2 = CellClassLevel(sf_cov_center=np.array([[0.03, 0.05], [-0.05, 0.12]]), class_level_id=6,
                                     sf_cov_surround=np.array([[0.07, 0.05], [-0.05, 0.30]]),
                                     sf_weight_surround=0.7, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class6_layout3 = CellClassLevel(sf_cov_center=np.array([[0.03, 0.05], [-0.05, 0.12]]), class_level_id=6,
                                     sf_cov_surround=np.array([[0.07, 0.05], [-0.05, 0.30]]),
                                     sf_weight_surround=0.7, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class6_layout4 = CellClassLevel(sf_cov_center=np.array([[0.03, 0.05], [-0.05, 0.12]]), class_level_id=6,
                                     sf_cov_surround=np.array([[0.07, 0.05], [-0.05, 0.30]]),
                                     sf_weight_surround=0.7, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))
cell_class6_layout5 = CellClassLevel(sf_cov_center=np.array([[0.03, 0.05], [-0.05, 0.12]]), class_level_id=6,
                                     sf_cov_surround=np.array([[0.07, 0.05], [-0.05, 0.30]]),
                                     sf_weight_surround=0.7, num_cells=6, xlim=(-0.5, 0.5), ylim=(-0.6, 0.6))


# Create experimental level with cell classes
experimental = ExperimentalLevel(tf_weight_surround=0.2, tf_sigma_center=0.05,
                                 tf_sigma_surround=0.12, tf_mean_center=0.08,
                                 tf_mean_surround=0.12, tf_weight_center=1,
                                 tf_offset=0, cell_classes=[cell_class1_layout1, cell_class2_layout1,
                                                            cell_class3_layout1, cell_class4_layout1,
                                                            cell_class5_layout1, cell_class6_layout1])

experimental2 = ExperimentalLevel(tf_weight_surround=0.3, tf_sigma_center=0.04,
                                  tf_sigma_surround=0.10, tf_mean_center=0.07,
                                  tf_mean_surround=0.10, tf_weight_center=1,
                                  tf_offset=0, cell_classes=[cell_class1_layout2, cell_class2_layout2,
                                                             cell_class3_layout2, cell_class4_layout2,
                                                             cell_class5_layout2, cell_class6_layout2])

experimental3 = ExperimentalLevel(tf_weight_surround=0.4, tf_sigma_center=0.03,
                                  tf_sigma_surround=0.09, tf_mean_center=0.06,
                                  tf_mean_surround=0.11, tf_weight_center=1,
                                  tf_offset=0, cell_classes=[cell_class1_layout3, cell_class2_layout3,
                                                             cell_class3_layout3, cell_class4_layout3,
                                                             cell_class5_layout3, cell_class6_layout3])

experimental4 = ExperimentalLevel(tf_weight_surround=0.1, tf_sigma_center=0.08,
                                  tf_sigma_surround=0.2, tf_mean_center=0.15,
                                  tf_mean_surround=0.2, tf_weight_center=1,
                                  tf_offset=0, cell_classes=[cell_class1_layout4, cell_class2_layout4,
                                                             cell_class3_layout4, cell_class4_layout4,
                                                             cell_class5_layout4, cell_class6_layout4])

experimental5 = ExperimentalLevel(tf_weight_surround=-0.1, tf_sigma_center=0.07,
                                  tf_sigma_surround=0.18, tf_mean_center=0.13,
                                  tf_mean_surround=0.18, tf_weight_center=-1,
                                  tf_offset=0, cell_classes=[cell_class1_layout5, cell_class2_layout5,
                                                             cell_class3_layout5, cell_class4_layout5,
                                                             cell_class5_layout5, cell_class6_layout5])