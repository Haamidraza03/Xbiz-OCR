import { useState } from 'react';
import { Card, Form, Button, Row, Col } from 'react-bootstrap';
import SuccessModal from './SuccessModal';
import { useDeployments } from '../contexts/DeploymentContext';

const workloads = [
  "web-frontend", "api-gateway", "user-service", "payment-service",
  "notification-service", "analytics-engine", "database-service"
];

const developers = [
  "John Smith", "Sarah Johnson", "Mike Chen", "Emily Davis",
  "David Wilson", "Lisa Anderson", "Ryan Martinez"
];

const deployers = [
  "DevOps Team", "Release Manager", "Senior Developer",
  "System Administrator", "CI/CD Pipeline"
];

const createNewEntry = () => ({
  environment: '',
  workload: '',
  version: '',
  changes_made: '',
  config_changes: '',
  changes_done_by: '',
  deployed_by: '',
});

const DeploymentForm = () => {
  const { addDeployments } = useDeployments();
  const [entries, setEntries] = useState([createNewEntry()]);
  const [showModal, setShowModal] = useState(false);
  const [message, setMessage] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const addEntry = () => {
    setEntries([...entries, createNewEntry()]);
  };

  const removeEntry = (index) => {
    const newEntries = entries.filter((_, i) => i !== index);
    setEntries(newEntries);
  };

  const updateEntry = (index, field, value) => {
    const newEntries = [...entries];
    newEntries[index] = { ...newEntries[index], [field]: value };
    setEntries(newEntries);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    setSubmitting(true);

    // Simulate API delay (optional, but matches async feel)
    setTimeout(() => {
      const created = addDeployments(entries);
      setMessage(`${created.length} deployment(s) created successfully`);
      setShowModal(true);
      setEntries([createNewEntry()]); // Reset to one entry
      setSubmitting(false);
    }, 500);
  };

  return (
    <div className="row">
      <div className="col-md-12">
        <Card className="shadow">
          <Card.Header className="bg-success text-white">
            <h4 className="card-title mb-0">
              <i className="fas fa-plus-circle me-2"></i>Create New Deployment
            </h4>
          </Card.Header>
          <Card.Body>
            <Form id="deploymentForm" onSubmit={handleSubmit}>
              <div id="deploymentEntries">
                {entries.map((entry, index) => (
                  <div key={index} className="deployment-entry border rounded p-4 mb-4 bg-light">
                    <div className="d-flex justify-content-between align-items-center mb-3">
                      <h5 className="text-primary mb-0">
                        <i className="fas fa-cog me-2"></i>Deployment Entry #{index + 1}
                      </h5>
                      {index > 0 && (
                        <Button
                          variant="outline-danger"
                          size="sm"
                          onClick={() => removeEntry(index)}
                        >
                          <i className="fas fa-trash"></i> Remove
                        </Button>
                      )}
                    </div>
                    <Row>
                      <Col md={6} className="mb-3">
                        <Form.Label>Environment <span className="text-danger">*</span></Form.Label>
                        <Form.Select
                          value={entry.environment}
                          onChange={(e) => updateEntry(index, 'environment', e.target.value)}
                          required
                        >
                          <option value="">Select Environment</option>
                          <option value="uat">UAT</option>
                          <option value="preprod">Pre-Production</option>
                          <option value="prod">Production</option>
                        </Form.Select>
                      </Col>
                      <Col md={6} className="mb-3">
                        <Form.Label>Workload <span className="text-danger">*</span></Form.Label>
                        <Form.Select
                          value={entry.workload}
                          onChange={(e) => updateEntry(index, 'workload', e.target.value)}
                          required
                        >
                          <option value="">Select Workload</option>
                          {workloads.map((w) => (
                            <option key={w} value={w}>{w}</option>
                          ))}
                        </Form.Select>
                      </Col>
                    </Row>
                    <Row>
                      <Col md={6} className="mb-3">
                        <Form.Label>Version <span className="text-danger">*</span></Form.Label>
                        <Form.Control
                          type="text"
                          value={entry.version}
                          onChange={(e) => updateEntry(index, 'version', e.target.value)}
                          placeholder="e.g., v1.2.3"
                          required
                        />
                      </Col>
                      <Col md={6} className="mb-3">
                        <Form.Label>Changes Done By <span className="text-danger">*</span></Form.Label>
                        <Form.Select
                          value={entry.changes_done_by}
                          onChange={(e) => updateEntry(index, 'changes_done_by', e.target.value)}
                          required
                        >
                          <option value="">Select Developer</option>
                          {developers.map((d) => (
                            <option key={d} value={d}>{d}</option>
                          ))}
                        </Form.Select>
                      </Col>
                    </Row>
                    <Row>
                      <Col md={12} className="mb-3">
                        <Form.Label>Deployed By <span className="text-danger">*</span></Form.Label>
                        <Form.Select
                          value={entry.deployed_by}
                          onChange={(e) => updateEntry(index, 'deployed_by', e.target.value)}
                          required
                        >
                          <option value="">Select Deployer</option>
                          {deployers.map((d) => (
                            <option key={d} value={d}>{d}</option>
                          ))}
                        </Form.Select>
                      </Col>
                    </Row>
                    <Row>
                      <Col md={6} className="mb-3">
                        <Form.Label>Changes Made <span className="text-danger">*</span></Form.Label>
                        <Form.Control
                          as="textarea"
                          rows={4}
                          value={entry.changes_made}
                          onChange={(e) => updateEntry(index, 'changes_made', e.target.value)}
                          placeholder="Describe the changes made in this deployment..."
                          required
                        />
                      </Col>
                      <Col md={6} className="mb-3">
                        <Form.Label>Config Changes <span className="text-danger">*</span></Form.Label>
                        <Form.Control
                          as="textarea"
                          rows={4}
                          value={entry.config_changes}
                          onChange={(e) => updateEntry(index, 'config_changes', e.target.value)}
                          placeholder="Describe any configuration changes..."
                          required
                        />
                      </Col>
                    </Row>
                  </div>
                ))}
              </div>
              <div className="text-center mb-4">
                <Button type="button" variant="outline-primary" onClick={addEntry}>
                  <i className="fas fa-plus me-2"></i>Add Another Deployment
                </Button>
              </div>
              <div className="text-center">
                <Button type="submit" variant="success" size="lg" disabled={submitting}>
                  {submitting ? (
                    <><i className="fas fa-spinner fa-spin me-2"></i>Submitting...</>
                  ) : (
                    <><i className="fas fa-rocket me-2"></i>Deploy All</>
                  )}
                </Button>
              </div>
            </Form>
          </Card.Body>
        </Card>
      </div>
      <SuccessModal show={showModal} onHide={() => setShowModal(false)} message={message} />
    </div>
  );
};

export default DeploymentForm;