#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>

#include "MNN_generated.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: patch_mnn_wrapper_nchw input.mnn output.mnn\n";
        return 1;
    }
    std::ifstream in(argv[1], std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(in)), {});
    if (data.empty()) return 1;
    std::unique_ptr<MNN::NetT> net(MNN::GetNet(data.data())->UnPack());
    int patched = 0;
    for (auto& op : net->oplists) {
        if (op->type != MNN::OpType_Input) continue;
        op->main.AsInput()->dformat = MNN::MNN_DATA_FORMAT_NCHW;
        ++patched;
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(MNN::Net::Pack(builder, net.get()));
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(builder.GetBufferPointer()), builder.GetSize());
    if (!out) return 1;
    std::cout << "Patched " << patched << " input tensor formats to NCHW\n";
}
