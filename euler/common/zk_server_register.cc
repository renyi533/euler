/* Copyright 2018 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "euler/common/zk_server_register.h"

#include <algorithm>
#include <vector>
#include <string>
#include <utility>
#include <assert.h>

#include "glog/logging.h"
#include "euler/common/string_util.h"
#include "euler/common/zk_util_cache.h"

namespace euler {
namespace common {

namespace {

std::string MetaToBytes(const Meta &meta, const Meta &shard_meta) {
  std::vector<std::string> lines;

  std::transform(
      meta.begin(), meta.end(), std::back_inserter(lines),
      [](const std::pair<std::string, std::string> &key_value) {
        return join_string({key_value.first, key_value.second}, ":");
      });
  std::transform(
      shard_meta.begin(), shard_meta.end(), std::back_inserter(lines),
      [](const std::pair<std::string, std::string> &key_value) {
        return join_string({"", key_value.first, key_value.second}, ":");
      });

  return join_string(lines, "\n");
}

std::string ShardToBytes(size_t shard_index, const Server server) {
  return join_string({std::to_string(shard_index), server}, "#");
}

void ZkLogCallback(const char * /*message*/) { }

}  // namespace

bool ZkServerRegister::Initialize() {
  {
    std::lock_guard<std::mutex> lock(zk_mu_);

    if (zk_handle_) {
      return true;
    }

    zhandle_t *zh = zookeeper_init2(zk_addr_.c_str(), Watcher, 60000, nullptr,
                                    this, 0, ZkLogCallback);
    if (zh == nullptr) {
      LOG(ERROR) << "Fail to initialize ZK connection.";
      assert(false);
      return false;
    }
    zk_handle_ = zh;
  }

  int rc = zoo_create(zk_handle_, zk_path_.c_str(), "", 0, &ZOO_OPEN_ACL_UNSAFE,
                      0, nullptr, 0);
  if (rc != ZOK && rc != ZNODEEXISTS) {
    LOG(ERROR) << "ZK error when creating root node: " << zerror(rc) << ".";
    assert(false);
  }

  return true;
}

ZkServerRegister::~ZkServerRegister() {
  std::lock_guard<std::mutex> lock(zk_mu_);
  zookeeper_close(zk_handle_);
  zk_handle_ = nullptr;
}

bool ZkServerRegister::RegisterShard(size_t shard_index, const Server &server,
                                     const Meta &meta, const Meta &shard_meta) {
  std::string shard_zk_child = ShardToBytes(shard_index, server);
  std::string shard_zk_path = join_string({zk_path_, shard_zk_child}, "/");
  std::string shard_meta_bytes = MetaToBytes(meta, shard_meta);

  int rc = zoo_create(zk_handle_, shard_zk_path.c_str(),
                      shard_meta_bytes.c_str(), shard_meta_bytes.size(),
                      &ZOO_OPEN_ACL_UNSAFE, ZOO_EPHEMERAL, nullptr, 0);
  if (rc != ZOK) {
    LOG(ERROR) << "ZK error when creating meta: " << zerror(rc) << ".";
    assert(false);
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    registered_.emplace(shard_zk_path, shard_meta_bytes);
  }
  return true;
}

bool ZkServerRegister::DeregisterShard(size_t shard_index,
                                       const Server &server) {
  std::string shard_zk_child = ShardToBytes(shard_index, server);
  std::string shard_zk_path = join_string({zk_path_, shard_zk_child}, "/");

  int rc = zoo_delete(zk_handle_, shard_zk_path.c_str(), -1);
  if (rc != ZOK) {
    LOG(ERROR) << "ZK error when deleting meta: " << zerror(rc) << ".";
    assert(false);
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(mu_);
    registered_.erase(shard_zk_child);
  }
  return true;
}

void ZkServerRegister::Watcher(zhandle_t *zh, int /*type*/, int state,
                               const char * /*path*/, void *data) {
  if (state == ZOO_EXPIRED_SESSION_STATE) {
    zookeeper_close(zh);

    ZkServerRegister *self = static_cast<ZkServerRegister *>(data);
    {
      std::lock_guard<std::mutex> lock(self->zk_mu_);

      self->zk_handle_ = nullptr;
      while (self->zk_handle_ == nullptr) {
        LOG(WARNING) << "Reconnecting ZK ...";
        self->zk_handle_ = zookeeper_init2(self->zk_addr_.c_str(), Watcher,
                                           60000, nullptr, self, 0,
                                           ZkLogCallback);
      }
    }

    {
      std::lock_guard<std::mutex> lock(self->mu_);

      for (const auto &registered : self->registered_) {
        const std::string &shard_zk_path = registered.first;
        const std::string &shard_meta_bytes = registered.second;
        int rc = zoo_create(self->zk_handle_, shard_zk_path.c_str(),
                            shard_meta_bytes.c_str(), shard_meta_bytes.size(),
                            &ZOO_OPEN_ACL_UNSAFE, ZOO_EPHEMERAL, nullptr, 0);
        if (rc != ZOK) {
          LOG(ERROR) << "ZK error when creating meta: " << zerror(rc) << ".";
          assert(false);
        }
      }
    }
  }
}

std::shared_ptr<ServerRegister> GetServerRegister(const std::string& zk_addr,
                                                  const std::string& zk_path) {
  return GetOrCreate<ZkServerRegister>(zk_addr, zk_path);
}

}  // namespace common
}  // namespace euler
