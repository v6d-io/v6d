/** Copyright 2020-2023 Alibaba Group Holding Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "server/util/kubectl.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/bind.hpp"     // IWYU pragma: keep
#include "boost/process.hpp"  // IWYU pragma: keep

#include "common/util/asio.h"  // IWYU pragma: keep
#include "common/util/env.h"
#include "common/util/logging.h"  // IWYU pragma: keep

namespace vineyard {

Kubectl::Kubectl(asio::io_context& context) : proc_(new Process(context)) {}

Kubectl::~Kubectl() { proc_->Terminate(); }

static std::string generate_local_object(
    std::map<InstanceID, std::string> const& instances, const json& object) {
  std::string object_id = object["id"].get_ref<std::string const&>();
  std::string signature =
      SignatureToString(object["signature"].get<Signature>());
  std::string type_name = object["typename"].get_ref<std::string const&>();

  std::string job_name = object.value("JOB_NAME", "\"\"");
  std::string client_pod_name = object.value("POD_NAME", "\"\"");
  std::string client_pod_namespace = object.value("POD_NAMESPACE", "\"\"");

  InstanceID instance_id = object["instance_id"].get<InstanceID>();
  std::string vineyardd_name = read_env("VINEYARDD_NAME");
  std::string namespace_ = read_env("VINEYARDD_NAMESPACE");
  std::string uid = read_env("VINEYARDD_UID");

  std::string hostname = "0.0.0.0";
  if (instances.find(instance_id) != instances.end()) {
    hostname = instances.at(instance_id);
  }

  /* clang-format off */
  std::string crd = "\n"
                    "\n---"
                    "\n"
                    "\napiVersion: k8s.v6d.io/v1alpha1"
                    "\nkind: LocalObject"
                    "\nmetadata:"
                    "\n  name: " + object_id +
                    "\n  namespace: " + namespace_ +
                    "\n  labels:"
                    "\n    k8s.v6d.io/signature: " + signature +
                    "\n    k8s.v6d.io/job: " + job_name +
                    "\n    k8s.v6d.io/created-podname: " + client_pod_name +
                    "\n    k8s.v6d.io/created-podnamespace: "
                    + client_pod_namespace +
                    "\n  ownerReferences:"
                    "\n    - apiVersion: k8s.v6d.io/v1alpha1"
                    "\n      kind: Vineyardd"
                    "\n      name: " + vineyardd_name +
                    "\n      uid: " + uid +
                    "\nspec:"
                    "\n  id: " + object_id +
                    "\n  signature: " + signature +
                    "\n  typename: " + type_name +
                    "\n  instance_id: " + std::to_string(instance_id) +
                    "\n  hostname: " + hostname +
                    "\n  metadata: " + type_name +
                    "\n\n";
  /* clang-format on */
  return crd;
}

static std::string generate_global_object(
    std::map<InstanceID, std::string> const& instances, const json& object) {
  std::string object_id = object["id"].get_ref<std::string const&>();
  std::string signature =
      SignatureToString(object["signature"].get<Signature>());
  std::string type_name = object["typename"].get_ref<std::string const&>();
  std::vector<std::string> members;

  std::string job_name = object.value("JOB_NAME", "\"\"");
  std::string client_pod_name = object.value("POD_NAME", "\"\"");
  std::string client_pod_namespace = object.value("POD_NAMESPACE", "\"\"");

  std::string vineyardd_name = read_env("VINEYARDD_NAME");
  std::string namespace_ = read_env("VINEYARDD_NAMESPACE");
  std::string uid = read_env("VINEYARDD_UID");

  std::string crds;
  for (auto const& kv : object.items()) {
    if (kv.value().is_object() && !kv.value().empty()) {
      crds += generate_local_object(instances, kv.value());
      members.emplace_back(
          SignatureToString(kv.value()["signature"].get<Signature>()));
    }
  }

  /* clang-format off */
  std::string crd = "\n"
                    "\n---"
                    "\n"
                    "apiVersion: k8s.v6d.io/v1alpha1"
                    "\nkind: GlobalObject"
                    "\nmetadata:"
                    "\n  name: " + object_id +
                    "\n  namespace: " + namespace_ +
                    "\n  labels:"
                    "\n    k8s.v6d.io/job: " + job_name +
                    "\n    k8s.v6d.io/created-podname: " + client_pod_name +
                    "\n    k8s.v6d.io/created-podnamespace: "
                    + client_pod_namespace +
                    "\n  ownerReferences:"
                    "\n    - apiVersion: k8s.v6d.io/v1alpha1"
                    "\n      kind: Vineyardd"
                    "\n      name: " + vineyardd_name +
                    "\n      uid: " + uid +
                    "\nspec:"
                    "\n  id: " + object_id +
                    "\n  signature: " + signature +
                    "\n  typename: " + type_name +
                    "\n  metadata: " + type_name +
                    "\n  members:";
  /* clang-format on */

  for (auto const& sig : members) {
    crd += "\n  - " + sig;
  }
  crd += "\n";
  return crd + crds;
}

void Kubectl::CreateObject(const json& cluster_meta, const json& object) {
  std::map<InstanceID, std::string> instances;
  for (auto const& kv : cluster_meta.items()) {
    InstanceID instance_id;
    std::stringstream(kv.key().substr(1)) >> instance_id;
    instances.emplace(instance_id,
                      kv.value()["nodename"].get_ref<std::string const&>());
  }

  std::string crds;
  if (object.value("global", false)) {
    crds = generate_global_object(instances, object);
  } else {
    crds = generate_local_object(instances, object);
  }
  VLOG(10) << "Apply CRDs: " << crds;
  this->Create(crds, [](const Status& status) { return status; });
}

void Kubectl::DeleteObject(const json& object) {
  std::map<InstanceID, std::string> instances;
  std::string crds;
  if (object.value("global", false)) {
    crds = generate_global_object(instances, object);
  } else {
    crds = generate_local_object(instances, object);
  }
  VLOG(10) << "Deleting CRDs: " << crds;
  this->Delete(crds, [](const Status& status) { return status; });
}

void Kubectl::Create(const std::string& content, callback_t<> callback) {
  proc_->Start("kubectl", {"apply", "-f", "-"},
               [](Status const&, const std::string&) { return Status::OK(); });
  proc_->AsyncWrite(content, [this, callback](Status const& status) {
    // TODO: improve the error diagnostic
    if (!status.ok()) {
      for (auto const& line : Diagnostic()) {
        VLOG(10) << "kubectl apply output: " << line;
      }
    }
    return callback(status);
  });
}

void Kubectl::Delete(const std::string& content, callback_t<> callback) {
  proc_->Start("kubectl", {"delete", "-f", "-"},
               [](Status const&, const std::string&) { return Status::OK(); });
  proc_->AsyncWrite(content, [this, callback](Status const& status) {
    // TODO: improve the error diagnostic
    if (!status.ok()) {
      for (auto const& line : Diagnostic()) {
        VLOG(10) << "kubectl delete output: " << line;
      }
    }
    return callback(status);
  });
}

void Kubectl::Finish() {
  proc_->Finish();
  for (auto const& line : Diagnostic()) {
    VLOG(10) << "kubectl output: " << line;
  }
  proc_->Wait();
  VLOG(10) << "kubectl exit with: " << proc_->ExitCode();
}

}  // namespace vineyard
