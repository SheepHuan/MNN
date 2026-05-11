//
//  PrefixCachePath.hpp
//  MNN
//
//  Shared helpers for the kvshare prefix cache directory layout.
//

#ifndef PrefixCachePath_hpp
#define PrefixCachePath_hpp

#include "MNNFileUtils.h"

#include <string>

namespace MNN {

inline std::string prefixCacheObjectsDir(const std::string& root) {
    return MNNFilePathConcat(root, "objects");
}

inline std::string prefixCacheBackendDir(const std::string& root, const std::string& backend) {
    return MNNFilePathConcat(prefixCacheObjectsDir(root), backend);
}

inline std::string prefixCacheObjectDir(const std::string& root, const std::string& backend,
                                        const std::string& cacheName) {
    return MNNFilePathConcat(prefixCacheBackendDir(root, backend), cacheName);
}

inline std::string prefixCacheLayerDir(const std::string& root, const std::string& backend,
                                       const std::string& cacheName) {
    return MNNFilePathConcat(prefixCacheObjectDir(root, backend, cacheName), "layers");
}

inline std::string prefixCacheLayerBase(const std::string& root, const std::string& backend,
                                        const std::string& cacheName, int layer) {
    return MNNFilePathConcat(prefixCacheLayerDir(root, backend, cacheName), std::to_string(layer));
}

inline std::string prefixCacheMetaPath(const std::string& root, const std::string& backend,
                                       const std::string& cacheName) {
    return MNNFilePathConcat(prefixCacheObjectDir(root, backend, cacheName), "meta.json");
}

inline std::string prefixCacheTokensPath(const std::string& root, const std::string& backend,
                                         const std::string& cacheName) {
    return MNNFilePathConcat(prefixCacheObjectDir(root, backend, cacheName), "tokens.json");
}

inline bool ensurePrefixCacheObjectDirs(const std::string& root, const std::string& backend,
                                        const std::string& cacheName) {
    if (!MNNCreateDir(root.c_str())) {
        return false;
    }
    auto objectsDir = prefixCacheObjectsDir(root);
    if (!MNNCreateDir(objectsDir.c_str())) {
        return false;
    }
    auto backendDir = prefixCacheBackendDir(root, backend);
    if (!MNNCreateDir(backendDir.c_str())) {
        return false;
    }
    auto objectDir = prefixCacheObjectDir(root, backend, cacheName);
    if (!MNNCreateDir(objectDir.c_str())) {
        return false;
    }
    auto layerDir = prefixCacheLayerDir(root, backend, cacheName);
    return MNNCreateDir(layerDir.c_str());
}

} // namespace MNN

#endif // PrefixCachePath_hpp
