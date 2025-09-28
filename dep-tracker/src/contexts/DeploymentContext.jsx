import { createContext, useState, useContext } from 'react';
import { v4 as uuidv4 } from 'uuid';

const DeploymentContext = createContext();

export const DeploymentProvider = ({ children }) => {
  const initialDeployments = [
    {
      id: uuidv4(),
      created_at: '2024-09-20 14:30:00',
      environment: 'prod',
      workload: 'web-frontend',
      version: 'v2.1.5',
      changes_made: 'Fixed critical bug in user authentication flow',
      config_changes: 'Updated API timeout from 30s to 45s',
      changes_done_by: 'John Smith',
      deployed_by: 'DevOps Team',
    },
    {
      id: uuidv4(),
      created_at: '2024-09-21 09:15:00',
      environment: 'uat',
      workload: 'payment-service',
      version: 'v1.3.2',
      changes_made: 'Added new payment gateway integration\nImplemented retry mechanism for failed payments',
      config_changes: 'Added new environment variables for payment gateway',
      changes_done_by: 'Sarah Johnson',
      deployed_by: 'Release Manager',
    },
    {
      id: uuidv4(),
      created_at: '2024-09-22 16:45:00',
      environment: 'preprod',
      workload: 'user-service',
      version: 'v3.0.1',
      changes_made: 'Performance optimization for user profile queries',
      config_changes: 'Increased database connection pool size',
      changes_done_by: 'Mike Chen',
      deployed_by: 'Senior Developer',
    },
  ];

  const [deployments, setDeployments] = useState(initialDeployments);

  const addDeployments = (newDeployments) => {
    const created = newDeployments.map((dep) => ({
      ...dep,
      id: uuidv4(),
      created_at: new Date().toISOString().slice(0, 19).replace('T', ' '),
    }));
    setDeployments((prev) => [...created, ...prev]);
    return created;
  };

  return (
    <DeploymentContext.Provider value={{ deployments, addDeployments }}>
      {children}
    </DeploymentContext.Provider>
  );
};

export const useDeployments = () => useContext(DeploymentContext);