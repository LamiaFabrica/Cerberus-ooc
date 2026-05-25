#include "hq/cerberus_slipstream.hpp"
int main() {
    using namespace hq::cerberus::slipstream;
    auto msg = SlipstreamMessage::create(SlipstreamMessageType::CMD_EXECUTE, {}, 0);
    return 0;
}
