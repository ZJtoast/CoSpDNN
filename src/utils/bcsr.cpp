
class BCSRCSCMatrix : public SparseMatrix
{
public:
    int blocksize_r, blocksize_c;
    int bcsr_block_num, bcsc_block_num;
    std::vector<float> bcsr_values_;
    std::vector<float> bcsc_values_;
    std::vector<int> bcsr_col_index_; // index: the start col idx in bcsr
    std::vector<int> bcsc_row_index_;
    std::vector<int> bcsr_len_; // len: the row_ptr in bcsr
    std::vector<int> bcsc_len_;
    std::vector<int> bcsr_row_index_;
    std::vector<int> bcsc_col_index_;

    void coo2bcsr(COOMatrix &coo_matrix)
    {
        auto iter = coo_matrix.begin();
        coo_matrix.to_row_first_ordered();
        int start_r = ((*iter).row);
        int start_c = ((*iter).col);
        bcsr_len_.push_back(0);
        // initialize
        bcsr_row_index_.push_back(start_r);
        bcsr_col_index_.push_back(start_c);
        bcsr_values_.push_back((*iter).val);
        fill_n(back_inserter(bcsr_values_), blocksize_c * blocksize_r - 1, 0);
#ifdef MAT_CONVERT_TEST
        std::cout << start_c << std::endl;
#endif

        // for every nnz element:
        // first check whether is the start element
        // if not, find its position
        ++iter;
        int blk_row_ptr = 0;
        for (; iter != coo_matrix.end(); ++iter)
        {
            // check whether the start element
            int blk_row = bcsr_row_index_[blk_row_ptr];
            int cur_row = (*iter).row;
            int cur_col = (*iter).col;

            // within the range of current blk row
            if ((cur_row - blk_row) < blocksize_r)
            {
                bool find_my_blk = false;
                // search from last
                int i, blk_col;
                for (i = bcsr_col_index_.size() - 1; i >= bcsr_len_[blk_row_ptr]; i--)
                {
                    blk_col = bcsr_col_index_[i];
                    if (cur_col >= blk_col && cur_col - blk_col < blocksize_c)
                    {
                        find_my_blk = true;
                        break;
                    }
                }
                if (find_my_blk) // insert new element
                {
                    int blk_pos = i * blocksize_c * blocksize_r;
                    int cur_row_pos = cur_row - blk_row;
                    int cur_col_pos = cur_col - bcsr_col_index_[i];
#ifdef MAT_CONVERT_TEST
// std::cout << "blk_pos is " << blk_pos << "(my_row, my_col) is" << cur_row_pos << " " << cur_col_pos <<std::endl;
#endif
                    bcsr_values_[blk_pos + (cur_row_pos)*blocksize_r + cur_col_pos] = (*iter).val;
                }
                else // a new block
                {
#ifdef MAT_CONVERT_TEST
// std::cout << "cur_row is " << cur_row << "cur_col is " << cur_col << "i is " << i << "size of bcsr_col" << bcsr_col_index_.size() << "capacity of bcsr_col" << bcsr_col_index_.capacity()<< std::endl;
#endif
                    bcsr_col_index_.push_back(cur_col);
                    bcsr_values_.push_back((*iter).val);
                    fill_n(back_inserter(bcsr_values_), blocksize_c * blocksize_r - 1, 0);
                }
            }
            else // new row, new block
            {
#ifdef MAT_CONVERT_TEST
// std::cout << "cur_row is " << cur_row << "cur_col is " << cur_col << "blk_row_ptr is " << blk_row_ptr << std::endl;
#endif
                bcsr_row_index_.push_back(cur_row);
                blk_row_ptr++;
                bcsr_col_index_.push_back(cur_col);
                bcsr_len_.push_back(bcsr_col_index_.size() - 1);
                bcsr_values_.push_back((*iter).val);
                fill_n(back_inserter(bcsr_values_), blocksize_c * blocksize_r - 1, 0);
            }
        }

        // final update
        bcsr_len_.push_back(bcsr_col_index_.size());
        bcsr_block_num = bcsr_len_.size() - 1;
        return;
    }

    void coo2bcsc(COOMatrix &coo_matrix)
    {
        auto iter = coo_matrix.begin();
        coo_matrix.to_col_first_ordered();
        int start_r = ((*iter).row);
        int start_c = ((*iter).col);
        bcsc_len_.push_back(0);
        // initialize
        bcsc_col_index_.push_back(start_c);
        bcsc_row_index_.push_back(start_r);
        bcsc_values_.push_back((*iter).val);
        fill_n(back_inserter(bcsc_values_), blocksize_c * blocksize_r - 1, 0);

        // for every nnz element:
        // first check whether is the start element
        // if not, find its position
        ++iter;
        int blk_col_ptr = 0;
        for (; iter != coo_matrix.end(); ++iter)
        {
            // check whether the start element
            int blk_col = bcsc_col_index_[blk_col_ptr];
            int cur_row = (*iter).row;
            int cur_col = (*iter).col;

            // within the range of current blk row
            if ((cur_col - blk_col) < blocksize_c)
            {
                bool find_my_blk = false;
                // search from last
                int i, blk_row;
                for (i = bcsc_row_index_.size() - 1; i >= bcsc_len_[blk_col_ptr]; i--)
                {
                    blk_row = bcsc_row_index_[i];
                    if (cur_row >= blk_row && cur_row - blk_row < blocksize_r)
                    {
                        find_my_blk = true;
                        break;
                    }
                }
                if (find_my_blk) // insert new element
                {
                    int blk_pos = i * blocksize_c * blocksize_r;
                    int cur_col_pos = cur_col - blk_col;
                    int cur_row_pos = cur_row - bcsc_row_index_[i];
                    bcsc_values_[blk_pos + (cur_col_pos)*blocksize_c + cur_row_pos] = (*iter).val;
                }
                else // a new block
                {
                    bcsc_row_index_.push_back(cur_row);
                    bcsc_values_.push_back((*iter).val);
                    fill_n(back_inserter(bcsc_values_), blocksize_c * blocksize_r - 1, 0);
                }
            }
            else // new col, new block
            {
                bcsc_col_index_.push_back(cur_col);
                blk_col_ptr++;
                bcsc_row_index_.push_back(cur_row);
                bcsc_len_.push_back(bcsc_row_index_.size() - 1);
                bcsc_values_.push_back((*iter).val);
                fill_n(back_inserter(bcsc_values_), blocksize_c * blocksize_r - 1, 0);
            }
        }

        // final update
        bcsc_len_.push_back(bcsc_row_index_.size());
        bcsc_block_num = bcsc_len_.size() - 1;
        return;
    }

    void transpose()
    {
        std::vector<SparseDataType> bcsr_values_tmp = bcsr_values_;
        bcsr_values_ = bcsc_values_;
        bcsc_values_ = bcsr_values_tmp;

        std::vector<int> bcsr_row_index_tmp = bcsr_row_index_;
        bcsr_row_index_ = bcsc_col_index_;
        bcsc_col_index_ = bcsr_row_index_tmp;

        std::vector<int> bcsr_col_index_tmp = bcsr_col_index_;
        bcsr_col_index_ = bcsc_row_index_;
        bcsc_row_index_ = bcsr_col_index_tmp;

        std::vector<int> bcsr_len_tmp = bcsr_len_;
        bcsr_len_ = bcsc_len_;
        bcsc_len_ = bcsr_len_tmp;

        int bcsr_block_num_tmp = bcsr_block_num;
        bcsr_block_num = bcsc_block_num;
        bcsc_block_num = bcsr_block_num;
        return;
    }

    void print_bcsr()
    {
        std::cout << "Print the Matrix in BCSR Format" << std::endl;
        for (int i = 0; i < bcsr_len_.size() - 1; ++i)
        {
            std::cout << "In Row " << bcsr_row_index_[i] << ": " << std::endl;
            int len = bcsr_len_[i + 1] - bcsr_len_[i];
            for (int j = 0; j < len; ++j)
            {
                std::cout << "Starting Col: " << bcsr_col_index_[bcsr_len_[i] + j] << std::endl;
                for (int k = 0; k < blocksize_r * blocksize_c; k++)
                {
                    std::cout << bcsr_values_[(bcsr_len_[i] + j) * blocksize_r * blocksize_c + k] << " ";
                }
                std::cout << std::endl;
            }
        }
        return;
    }

    void print_bcsc()
    {
        std::cout << "Print the Matrix in BCSC Format" << std::endl;
        for (int i = 0; i < bcsc_len_.size() - 1; ++i)
        {
            std::cout << "In Col " << bcsc_col_index_[i] << ": " << std::endl;
            int len = bcsc_len_[i + 1] - bcsc_len_[i];
            for (int j = 0; j < len; ++j)
            {
                std::cout << "Starting Row: " << bcsc_row_index_[bcsc_len_[i] + j] << std::endl;
                for (int k = 0; k < blocksize_r * blocksize_c; k++)
                {
                    std::cout << bcsc_values_[(bcsc_len_[i] + j) * blocksize_r * blocksize_c + k] << " ";
                }
                std::cout << std::endl;
            }
        }
        return;
    }

    void init_from_COO(COOMatrix &coo_matrix)
    {
        coo2bcsr(coo_matrix);
#ifdef MAT_CONVERT_TEST
        std::cout << "finish coo to bcsr" << std::endl;
#endif
        coo2bcsc(coo_matrix);
        return;
    }

    // constructor
    BCSRCSCMatrix(std::string &filename, int begin_node_idx,
                  int blocksize_r_tmp, int blocksize_c_tmp)
    {
        COOMatrix coo(filename, begin_node_idx);
        blocksize_r = blocksize_r_tmp;
        blocksize_c = blocksize_c_tmp;
        init_from_COO(coo);
        return;
    }

    BCSRCSCMatrix(COOMatrix &coo_matrix, int blocksize_r_tmp, int blocksize_c_tmp)
    {
        blocksize_r = blocksize_r_tmp;
        blocksize_c = blocksize_c_tmp;
        init_from_COO(coo_matrix);
        return;
    }
};